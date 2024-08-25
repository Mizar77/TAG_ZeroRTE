import json
import random
import math 
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List
from torch.nn import CrossEntropyLoss


import torch
from fire import Fire
from pydantic.main import BaseModel
from tqdm import tqdm

from generation import LabelConstraint, TripletSearchDecoder
from modeling import (NewRelationExtractor, RelationGenerator, RelationModel,
                      select_model)
from utils import (RelationSentence, WikiDataset, delete_checkpoints,
                   load_wiki_relation_map, mark_fewrel_entity)

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b


class Sentence(BaseModel):
    triplets: List[RelationSentence]

    @property
    def tokens(self) -> List[str]:
        return self.triplets[0].tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def assert_valid(self):
        assert len(self.tokens) > 0
        for t in self.triplets:
            assert t.text == self.text
            assert len(t.head) > 0
            assert len(t.tail) > 0
            assert len(t.label) > 0


class Dataset(BaseModel):
    sents: List[Sentence]

    def get_labels(self) -> List[str]:
        return sorted(set(t.label for s in self.sents for t in s.triplets))

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sents = [Sentence(**json.loads(line)) for line in f]
        return cls(sents=sents)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sents:
                f.write(s.json() + "\n")

    @classmethod
    def load_fewrel(cls, path: str, path_properties: str = "data/wiki_properties.csv"):
        relation_map = load_wiki_relation_map(path_properties)
        groups = {}

        with open(path) as f:
            for i, lst in tqdm(json.load(f).items()):
                for raw in lst:
                    head, tail = mark_fewrel_entity(raw)
                    t = RelationSentence(
                        tokens=raw["tokens"],
                        head=head,
                        tail=tail,
                        label=relation_map[i].pLabel,
                        label_id=i,
                    )
                    groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        return cls(sents=sents)

    @classmethod
    def load_wiki(cls, path: str, path_properties: str = "data/wiki_properties.csv"):
        relation_map = load_wiki_relation_map(path_properties)
        sents = []
        with open(path) as f:
            ds = WikiDataset(
                mode="train", data=json.load(f), pid2vec=None, property2idx=None
            )
            for i in tqdm(range(len(ds))):
                triplets = ds.load_edges(i)
                triplets = [t for t in triplets if t.label_id in relation_map.keys()]
                for t in triplets:
                    t.label = relation_map[t.label_id].pLabel
                if triplets:
                    # ZSBERT only includes first triplet in each sentence
                    for t in triplets:
                        t.zerorc_included = False
                    triplets[0].zerorc_included = True

                    s = Sentence(triplets=triplets)
                    sents.append(s)

        data = cls(sents=sents)
        counter = Counter(t.label for s in data.sents for t in s.triplets)
        threshold = sorted(counter.values())[-113]  # Based on ZSBERT data stats
        labels = [k for k, v in counter.items() if v >= threshold]
        data = data.filter_labels(labels)
        return data

    def filter_labels(self, labels: List[str]):
        label_set = set(labels)
        sents = []
        for s in self.sents:
            triplets = [t for t in s.triplets if t.label in label_set]
            if triplets:
                s = s.copy(deep=True)
                s.triplets = triplets
                sents.append(s)
        return Dataset(sents=sents)

    def train_test_split(self, test_size: int, random_seed: int, by_label: bool):
        random.seed(random_seed)

        if by_label:
            labels = self.get_labels()
            labels_test = random.sample(labels, k=test_size)
            labels_train = sorted(set(labels) - set(labels_test))
            sents_train = self.filter_labels(labels_train).sents
            sents_test = self.filter_labels(labels_test).sents
        else:
            sents_train = [s for s in self.sents]
            sents_test = random.sample(self.sents, k=test_size)

        banned = set(s.text for s in sents_test)  # Prevent sentence overlap
        sents_train = [s for s in sents_train if s.text not in banned]
        assert len(self.sents) == len(sents_train) + len(sents_test)
        return Dataset(sents=sents_train), Dataset(sents=sents_test)

    def analyze(self):
        info = dict(
            sents=len(self.sents),
            unique_texts=len(set(s.triplets[0].text for s in self.sents)),
            lengths=str(Counter(len(s.triplets) for s in self.sents)),
            labels=len(self.get_labels()),
        )
        print(json.dumps(info, indent=2))


def write_data_splits(
    path_in: str,
    mode: str,
    folder_out: str = "outputs/data/splits/zero_rte",
    num_dev_labels: int = 5,
    num_test_labels: List[int] = [5, 10, 15],
    seeds: List[int] = [0, 1, 2, 3, 4],
):
    for n in num_test_labels:
        for s in seeds:
            if mode == "fewrel":
                data = Dataset.load_fewrel(path_in)
            elif mode == "wiki":
                data = Dataset.load_wiki(path_in)
            else:
                raise ValueError()

            train, test = data.train_test_split(
                test_size=n, random_seed=s, by_label=True
            )
            train, dev = train.train_test_split(
                test_size=num_dev_labels, random_seed=s, by_label=True
            )
            del data

            for key, data in dict(train=train, dev=dev, test=test).items():
                name = f"unseen_{n}_seed_{s}"
                path = Path(folder_out) / Path(path_in).stem / name / f"{key}.jsonl"
                data.save(str(path))
                print(dict(key=key, labels=len(data.get_labels()), path=path))


class Generator(BaseModel):
    load_dir: str
    save_dir: str
    num_gen_per_label: int = 250
    model_name: str = "generate"
    encoder_name: str = "generate"
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: Dataset, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.txt"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = model.get_encoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str):
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        data_train = Dataset.load(path_train)
        data_dev = Dataset.load(path_dev)
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def generate(self, labels: List[str], path_out: str):
        if Path(path_out).exists():
            return

        model = self.get_model()
        pipe = model.make_pipe()
        groups = {}
        assert isinstance(model, RelationGenerator)
        for relation in tqdm(labels, desc='Generating'):
            triplets, raw = model.generate(relation, self.num_gen_per_label, pipe=pipe)
            for t in triplets:
                groups.setdefault(t.text, []).append(t)

        sents = [Sentence(triplets=lst) for lst in groups.values()]
        data = Dataset(sents=sents)
        data.save(path_out)

    def estimate(self, path_in, path_out):
        # if Path(path_out).exists():
        #     return
        model = self.get_model()
        gpt_model = AutoModelForCausalLM.from_pretrained(model.model_dir).to('cuda')
        gpt_tokenizer = AutoTokenizer.from_pretrained(model.model_dir)
        encoder = model.get_encoder()
        data_in = Dataset.load(path_in)
        dataset = []
        for ins_i, ins in enumerate(data_in.sents):
            for tri_i, tri in enumerate(ins.triplets):
                if tri.generator_nll != 0.:
                    return 
                x, y = encoder.encode(tri)
                prefix = gpt_tokenizer.encode(x)
                dataset.append({'ins_i': ins_i, 'tri_i': tri_i, 'x': x, 'y': y, 'prefix_len': len(prefix) - 1})
        num_batch = math.ceil(len(dataset) / model.batch_size)
        bsz = model.batch_size
        dataset_counter = 0 
        for i in tqdm(range(num_batch), desc='generator estimating'):
            batch = dataset[i * bsz: (i + 1) * bsz]
            batch_input_text = [t['y'] for t in batch]
            inputs = gpt_tokenizer(batch_input_text, return_tensors='pt', padding=True)
            input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')
            labels = torch.where(input_ids == gpt_tokenizer.pad_token_id, -100, input_ids).to('cuda')
            outputs = gpt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            # shift 
            outputs = outputs[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            for o, l in zip(outputs, labels):
                loss = loss_fct(o.view(-1, o.shape[-1]), l.view(-1), )
                cond_o, cond_l = o[dataset[dataset_counter]['prefix_len']:], l[dataset[dataset_counter]['prefix_len']:]
                cond_loss = loss_fct(cond_o.view(-1, cond_o.shape[-1]), cond_l.view(-1))
                dataset[dataset_counter]['generator_nll'] = loss.item()
                dataset[dataset_counter]['cond_generator_nll'] = cond_loss.item()
                dataset_counter += 1
        assert dataset_counter == len(dataset) 
        for data in dataset:
            ins_i, tri_i, g_nll, cond_g_nll = data['ins_i'], data['tri_i'], data['generator_nll'], data['cond_generator_nll']
            assert ins_i < len(data_in.sents) and tri_i < len(data_in.sents[ins_i].triplets)
            tri = data_in.sents[ins_i].triplets[tri_i]
            tri.generator_nll = g_nll
            tri.cond_generator_nll = cond_g_nll
        data_in.save(path_out)

class Extractor(BaseModel):
    load_dir: str
    save_dir: str
    model_name: str = "new_extract"
    encoder_name: str = "extract"
    search_threshold: float = -0.9906
    model_kwargs: dict = {}

    def get_model(self) -> RelationModel:
        model = select_model(
            name=self.model_name,
            encoder_name=self.encoder_name,
            model_dir=str(Path(self.save_dir) / "model"),
            model_name=self.load_dir,
            data_dir=str(Path(self.save_dir) / "data"),
            do_pretrain=False,
            **self.model_kwargs,
        )
        return model

    def write_data(self, data: Dataset, name: str) -> str:
        model = self.get_model()
        path_out = Path(model.data_dir) / f"{name}.json"
        path_out.parent.mkdir(exist_ok=True, parents=True)
        encoder = model.get_encoder()
        lines = [encoder.encode_to_line(t) for s in data.sents for t in s.triplets]
        random.seed(model.random_seed)
        random.shuffle(lines)
        with open(path_out, "w") as f:
            f.write("".join(lines))
        return str(path_out)

    def fit(self, path_train: str, path_dev: str):
        model = self.get_model()
        if Path(model.model_dir).exists():
            return

        data_train = Dataset.load(path_train)
        data_train = Dataset.load(path_train)
        data_dev = Dataset.load(path_dev)
        path_train = self.write_data(data_train, "train")
        path_dev = self.write_data(data_dev, "dev")
        model.fit(path_train=path_train, path_dev=path_dev)
        delete_checkpoints(model.model_dir)

    def predict(self, path_in: str, path_out: str, use_label_constraint: bool = True):
        if Path(path_out).exists():
            return
        data = Dataset.load(path_in)
        texts = [s.text for s in data.sents]
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
        gen = model.load_generator(torch.device("cuda"))
        encoder = model.get_encoder()
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        sents = []

        for i in tqdm(range(0, len(texts), model.batch_size)):
            batch = texts[i : i + model.batch_size]
            x = [encoder.encode_x(t) for t in batch]
            outputs = model.gen_texts(
                x, gen, num_beams=1, save_scores=use_label_constraint
            )
            assert len(outputs) == len(x)

            for i, raw in enumerate(outputs):
                triplet = encoder.safe_decode(x[i], y=raw)
                if use_label_constraint:
                    assert gen.scores is not None
                    triplet = constraint.run(triplet, gen.scores[i])
                sents.append(Sentence(triplets=[triplet]))

        Dataset(sents=sents).save(path_out)

    def predict_multi(self, path_in: str, path_out: str):
        if Path(path_out).exists():
            return
        stem = Path(path_out).stem
        path_raw = path_out.replace(stem, f"{stem}_raw")
        print(dict(predict_multi=locals()))
        data = Dataset.load(path_in)
        model = self.get_model()
        assert isinstance(model, NewRelationExtractor)
        gen = model.load_generator(torch.device("cuda"))
        constraint = LabelConstraint(labels=data.get_labels(), tokenizer=gen.tokenizer)
        searcher = TripletSearchDecoder(
            gen=gen, encoder=model.get_encoder(), constraint=constraint
        )

        sents = [
            Sentence(tokens=s.tokens, triplets=searcher.run(s.text))
            for s in tqdm(data.sents)
        ]
        Dataset(sents=sents).save(path_raw)
        for s in sents:
            s.triplets = [t for t in s.triplets if t.score > self.search_threshold]
        Dataset(sents=sents).save(path_out)

    @staticmethod
    def score(path_pred: str, path_gold: str) -> dict:
        pred = Dataset.load(path_pred)
        gold = Dataset.load(path_gold)
        assert len(pred.sents) == len(gold.sents)
        num_pred = 0
        num_gold = 0
        num_correct = 0

        for i in range(len(gold.sents)):
            num_pred += len(pred.sents[i].triplets)
            num_gold += len(gold.sents[i].triplets)
            for p in pred.sents[i].triplets:
                for g in gold.sents[i].triplets:
                    if len(p.head) == 0 or len(p.tail) == 0:
                        num_pred -= 1 
                        continue
                    if (p.head, p.tail, p.label) == (g.head, g.tail, g.label):
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        info = dict(
            path_pred=path_pred,
            path_gold=path_gold,
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
        )
        return info
    def estimate(self, path_in, path_out):
        model = self.get_model()
        bart_model = AutoModelForSeq2SeqLM.from_pretrained(model.model_dir).to('cuda')
        bart_tokenizer = AutoTokenizer.from_pretrained(model.model_dir)
        encoder = model.get_encoder()
        data_in = Dataset.load(path_in)
        dataset = []
        for ins_i, ins in enumerate(data_in.sents):
            for tri_i, tri in enumerate(ins.triplets):
                if tri.extractor_nll != 0.:
                    return 
                x, y = encoder.encode(tri)
                dataset.append({'ins_i': ins_i, 'tri_i': tri_i, 'x': x, 'y': y})
        num_batch = math.ceil(len(dataset) / model.batch_size)
        bsz = model.batch_size
        dataset_counter = 0 
        for i in tqdm(range(num_batch), desc='extractor estimating'):
            batch = dataset[i * bsz: (i + 1) * bsz]
            batch_input_text = [t['x'] for t in batch]
            batch_output_text = [t['y'] for t in batch]
            inputs = bart_tokenizer(batch_input_text, return_tensors='pt', padding=True)
            input_ids, attention_mask = inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda')
            labels = bart_tokenizer(batch_output_text, return_tensors='pt', padding=True)
            labels = labels['input_ids'].to('cuda')
            labels = torch.where(labels == bart_tokenizer.pad_token_id, -100, labels).to('cuda')
            outputs = bart_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            # no shift for bart model 

            loss_fct = CrossEntropyLoss()
            for o, l in zip(outputs, labels):
                loss = loss_fct(o.view(-1, o.shape[-1]), l.view(-1), )
                dataset[dataset_counter]['extractor_nll'] = loss.item()
                dataset_counter += 1
        assert dataset_counter == len(dataset) 
        for data in dataset:
            ins_i, tri_i, e_nll = data['ins_i'], data['tri_i'], data['extractor_nll']
            assert ins_i < len(data_in.sents) and tri_i < len(data_in.sents[ins_i].triplets)
            tri = data_in.sents[ins_i].triplets[tri_i]
            tri.extractor_nll = e_nll
        data_in.save(path_out)

def main(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    g_encoder_name: str,
):
    print(dict(main=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator"),
        encoder_name=g_encoder_name
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    path_synthetic = str(Path(save_dir) / "synthetic.jsonl")
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    generator.generate(labels_dev + labels_test, path_out=path_synthetic)

    extractor_final = Extractor(
        load_dir=str(Path(save_dir) / "extractor" / "model"),
        save_dir=str(Path(save_dir) / "extractor_final"),
    )
    extractor_final.fit(path_synthetic, path_dev)

    # path_pred = str(Path(save_dir) / "pred.jsonl")
    # extractor_final.predict(path_in=path_test, path_out=path_pred)
    # results = extractor_final.score(path_pred, path_test)
    # print(json.dumps(results, indent=2))
    # with open(Path(save_dir) / "results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    run_eval(path_model=str(Path(save_dir) / "extractor_final"), path_test=path_test, mode='single', is_eval=False)
    run_eval(path_model=str(Path(save_dir) / "extractor_final"), path_test=path_test, mode='multi', is_eval=False)

    # return results

def main_dpo(path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
):
    print(dict(main_dpo=locals()))
    generator = Generator(
        load_dir=str(Path(save_dir) / "generator_dpo" / 'model'),
        save_dir=str(Path(save_dir) / "generator_dpo"),
    )

    # generator.fit(path_train, path_dev)
    path_synthetic = str(Path(save_dir) / "generator_dpo" / "synthetic.jsonl")
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    generator.generate(labels_dev + labels_test, path_out=path_synthetic)

    extractor_final = Extractor(
        load_dir=str(Path(save_dir) / "extractor" / "model"),
        save_dir=str(Path(save_dir) / "extractor_final_dpo"),
    )
    extractor_final.fit(path_synthetic, path_dev)

    # path_pred = str(Path(save_dir) / "pred.jsonl")
    # extractor_final.predict(path_in=path_test, path_out=path_pred)
    # results = extractor_final.score(path_pred, path_test)
    # print(json.dumps(results, indent=2))
    # with open(Path(save_dir) / "results.json", "w") as f:
    #     json.dump(results, f, indent=2)
    run_eval(path_model=str(Path(save_dir) / "extractor_final_dpo"), path_test=path_test, mode='single', is_eval=False)
    run_eval(path_model=str(Path(save_dir) / "extractor_final_dpo"), path_test=path_test, mode='multi', is_eval=False)

    # return results


def main_pseudo(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    num_iter: int
):
    print(dict(main=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator/iter0"),
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor/iter0"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    for i in range(num_iter):
        path_synthetic = str(Path(save_dir) / "synthetic" / f"{i}.jsonl")
        # path_synthetic_generator = str(Path(save_dir) / "synthetic" / f"{i}_gen.jsonl")
        # path_synthetic_extractor = str(Path(save_dir) / "synthetic" / f"{i}_ext.jsonl")
        generator.generate(labels_dev + labels_test, path_out=path_synthetic)
        # extractor.estimate(path_synthetic, path_synthetic_extractor)
        # generator.estimate(path_synthetic, path_synthetic_generator)
        path_filtered = path_synthetic
        extractor = Extractor(
            load_dir=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "extractor" / f'iter{i+1}'),
        )
        generator = Generator(
            load_dir=str(Path(save_dir) / "generator" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "generator" / f'iter{i+1}'),
        )
        extractor.fit(path_filtered, path_dev)
        generator.fit(path_filtered, path_dev)

        # path_pred_dev = str(Path(save_dir) / "pred_dev" / f"{i}.jsonl")
        # path_pred_test = str(Path(save_dir) / "pred_test" / f"{i}.jsonl")
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_test, mode='single', is_eval=False)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_test, mode='multi', is_eval=False)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_dev, mode='single', is_eval=True)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_dev, mode='multi', is_eval=True)

def sort_data(data):
    groups = dict()
    gen_list = []
    ext_list = []
    for ins in data.sents:
        for tri in ins.triplets:
            gen_list.append(tri.cond_generator_nll)
            ext_list.append(tri.extractor_nll)
            groups.setdefault(tri.label, []).append(tri)
    gen_mean, gen_std = np.mean(gen_list), np.std(gen_list)
    ext_mean, ext_std = np.mean(ext_list), np.std(ext_list)
    std_func = lambda x, mean, std: ((x - mean) / std) if std != 0 else (x - mean) 
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: std_func(x.extractor_nll, ext_mean, ext_std) + std_func(x.cond_generator_nll, gen_mean, gen_std))
    return groups 
    
def get_embed(data, model, tokenizer, device='cuda', type='ave'):
    model = model.to(device)
    model.eval()
    ret_data = dict()
    for rel in tqdm(data, desc=f'Getting Embed'):
        batch_size = 16
        ret_embed = []
        num_batch = math.ceil(len(data[rel]) / batch_size)
        for i in range(num_batch):
            batch = data[rel][i * batch_size: (i + 1) * batch_size]
            text = [' '.join(ins.tokens) for ins in batch]
            inputs = tokenizer(text, return_tensors='pt', padding=True)
            batch_inputs = dict()
            for k in inputs:
                batch_inputs[k] = inputs[k].to(device)
            output = model(**batch_inputs, output_hidden_states=True).hidden_states
            att = batch_inputs['attention_mask']
            hidden_states = (sum(output) / len(output))
            if type == 'ave':
                hidden_states = hidden_states * torch.unsqueeze(att, dim=2).expand_as(hidden_states)
                embed = torch.mean(hidden_states, dim=1)
            elif type == 'last':
                length =  torch.sum(att, dim=-1)
                embed = hidden_states[range(hidden_states.shape[0]), length - 1, :]
            else:
                raise ValueError(f'wrong type value: {type}')
            ret_embed.extend(embed.cpu().tolist())
        ret_data[rel] = [(d, e) for d, e in zip(data[rel], ret_embed)]
        assert len(data[rel]) == len(ret_embed)
    return ret_data

def sampling(data, num_sample):
    print(dict(sampling={'num_sample': num_sample}))
    ret = dict()
    for rel in data:
        if len(data[rel]) == 0:
            ret[rel] = []
            continue 
        cluster = data[rel]
        sampled = cluster[: max(1, num_sample // 2)]
        unsampled = [x for x in cluster if x not in sampled] 
        while len(sampled) < num_sample:
            embed = np.array([x[1] for x in unsampled])
            samp_embed = np.array([x[1] for x in sampled])
            cent = np.mean(samp_embed, axis=0)
            unnorm_sim = np.dot(embed, cent)
            norm_embed = np.linalg.norm(embed, axis=1)
            norm_cent = np.linalg.norm(cent)
            norm_sim = unnorm_sim / norm_embed / norm_cent
            sel_idx = np.argmin(norm_sim)
            sampled.append(unsampled[sel_idx])
            unsampled = [unsampled[i] for i in range(len(unsampled)) if i != sel_idx]
        ret[rel] = [x[0] for x in sampled] 
    return ret 

def filter_data(path_pseudo, path_train, path_out, total_pseudo_per_label, pseudo_ratio, with_train, by_rel, model_name=None, diverse=False):
    print(dict(filter_data=locals()))
    if Path(path_out).exists():
        return 
    if diverse:
        assert model_name is not None 
        if 'generator' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif 'extractor' in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError(f'Wrong Model Name: {model_name}')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    assert by_rel == True
    train_data = Dataset.load(path_train)
    pseudo_data = Dataset.load(path_pseudo)
    num_pseudo = int(total_pseudo_per_label * pseudo_ratio * len(pseudo_data.get_labels()))
    num_train = int(total_pseudo_per_label * (1- pseudo_ratio) * len(pseudo_data.get_labels()))
    num_pseudo_per_label = int(num_pseudo / len(pseudo_data.get_labels()))
    num_train_per_label = int(num_train / len(train_data.get_labels()))
    print(dict(num_pseudo=num_pseudo, num_train=num_train, num_pseudo_per_label=num_pseudo_per_label, num_train_per_label=num_train_per_label))
    
    # 按relation分类，并排序
    train_data = sort_data(train_data)
    pseudo_data = sort_data(pseudo_data)
    groups = dict()
    if with_train:
        if diverse:
            for k in train_data:
                train_data[k] = train_data[k][: 2 * num_train_per_label]
            train_data_embed = get_embed(train_data, model, tokenizer)
            groups = sampling(train_data_embed, num_train_per_label)
            for k in pseudo_data:
                pseudo_data[k] = pseudo_data[k][: 2 * num_pseudo_per_label]
            pseudo_data_embed = get_embed(pseudo_data, model, tokenizer)
            groups.update(sampling(pseudo_data_embed, num_pseudo_per_label))
        else:
            for k in train_data:
                groups[k] = train_data[k][: num_train_per_label]
            for k in pseudo_data:
                groups[k] = pseudo_data[k][: num_pseudo_per_label]

    # 按text合并
    if with_train:
        assert set(groups.keys()) == set(train_data.keys()) | set(pseudo_data.keys())
    else:
        assert set(groups.keys()) == set(pseudo_data.keys())
    text_data = dict()
    for k in groups:
        for tri in groups[k]:
            text_data.setdefault(tri.text, []).append(tri)
    sents = [Sentence(triplets=lst) for lst in text_data.values()]
    print('num of filtered data:', len(sents))
    data = Dataset(sents=sents)
    data.save(path_out)

def RLfilter_data(path_pseudo, path_train, path_out, total_pseudo_per_label, pseudo_ratio, with_train, by_rel):
    print(dict(filter_data=locals()))
    if Path(path_out).exists():
        return 
    assert by_rel == True
    train_data = Dataset.load(path_train)
    pseudo_data = Dataset.load(path_pseudo)
    num_pseudo = int(total_pseudo_per_label * pseudo_ratio * len(pseudo_data.get_labels()))
    num_train = int(total_pseudo_per_label * (1- pseudo_ratio) * len(pseudo_data.get_labels()))
    num_pseudo_per_label = int(num_pseudo / len(pseudo_data.get_labels()))
    num_train_per_label = int(num_train / len(train_data.get_labels()))
    print(dict(num_pseudo=num_pseudo, num_train=num_train, num_pseudo_per_label=num_pseudo_per_label, num_train_per_label=num_train_per_label))
    
    # 按relation分类，并排序
    train_data = sort_data(train_data)
    pseudo_data = sort_data(pseudo_data)
    groups = dict()
    if with_train:
        for k in train_data:
            groups[k] = train_data[k][: num_train_per_label]
    for k in pseudo_data:
        groups[k] = pseudo_data[k][: num_pseudo_per_label]

    # 按text合并
    if with_train:
        assert set(groups.keys()) == set(train_data.keys()) | set(pseudo_data.keys())
    else:
        assert set(groups.keys()) == set(pseudo_data.keys())
    text_data = dict()
    for k in groups:
        for tri in groups[k]:
            text_data.setdefault(tri.text, []).append(tri)
    sents = [Sentence(triplets=lst) for lst in text_data.values()]
    print('num of filtered data:', len(sents))
    data = Dataset(sents=sents)
    data.save(path_out)

def main_dual(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    num_iter: int,
    with_train: bool,
    by_rel: bool,
    score_only_ext: bool = False, 
    limit: int = 5000,
    g_encoder_name: str = 'generate', 
    num_gen_per_label: int = 250, 
    diverse: bool = False, 
):
    print(dict(main=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator/iter0"),
        num_gen_per_label=num_gen_per_label if not diverse else num_gen_per_label * 2, 
        encoder_name=g_encoder_name,
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor/iter0"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    for i in range(num_iter):
        path_synthetic = str(Path(save_dir) / "synthetic" / f"{i}.jsonl")
        path_synthetic_generator = str(Path(save_dir) / "synthetic" / f"{i}_gen.jsonl")
        path_synthetic_extractor = str(Path(save_dir) / "synthetic" / f"{i}_ext.jsonl")
        generator.generate(labels_dev + labels_test, path_out=path_synthetic)
        if not score_only_ext:
            generator.estimate(path_synthetic, path_synthetic)
        extractor.estimate(path_synthetic, path_synthetic)
        

        # filter
        path_filtered = str(Path(save_dir) / "filtered" / f"{i}.jsonl")
        filter_data(path_synthetic, path_train, path_filtered, num_gen_per_label, (i + 1.) / num_iter, with_train, by_rel, model_name=str(Path(save_dir) / "generator" / f'iter{i}' / "model"), diverse=diverse)

        extractor = Extractor(
            load_dir=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "extractor" / f'iter{i+1}'),
        )
        generator = Generator(
            load_dir=str(Path(save_dir) / "generator" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "generator" / f'iter{i+1}'),
            num_gen_per_label=num_gen_per_label if not diverse else num_gen_per_label * 2, 
            encoder_name=g_encoder_name,
        )
        extractor.fit(path_filtered, path_dev)
        generator.fit(path_filtered, path_dev)

        # path_pred_dev = str(Path(save_dir) / "pred_dev" / f"{i}.jsonl")
        # path_pred_test = str(Path(save_dir) / "pred_test" / f"{i}.jsonl")
        # run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
        #          path_test=path_test, mode='single', is_eval=False, limit=limit)
        # run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
        #          path_test=path_test, mode='multi', is_eval=False, limit=limit)

        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_dev, mode='single', is_eval=True, limit=limit)
        # run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                #  path_test=path_dev, mode='multi', is_eval=True, limit=limit)
    
    eval_best(path_test=path_test, save_dir=save_dir, num_iter=num_iter, limit=limit)

def eval_best_loss(path_test: str, save_dir: str, num_iter: int, limit: int):
    print(dict(eval_best_loss=locals()))
    best = 100.
    best_i = -1
    for i in range(num_iter + 1):
        path_model = str(Path(save_dir) /'extractor' / f'iter{i}')
        path_results = str(Path(path_model) / 'model' / f"eval_results.json")
        with open(path_results) as f:
            data = json.load(f)
            metric = data['eval_loss']
            if metric < best:
                best = metric 
                best_i = i
    print('best eval metric is', best, 'at iter', best_i)
    path_model = str(Path(save_dir) / 'extractor' / f'iter{best_i}')
    run_eval(path_model=path_model, path_test=path_test, mode='single', is_eval=False)
    run_eval(path_model=path_model, path_test=path_test, mode='multi', is_eval=False)
    # run_eval(path_model=path_model, path_test=path_test, mode='all_single', is_eval=False)
    # run_eval(path_model=path_model, path_test=path_test, mode='all_multi', is_eval=False)


def main_dual_loss(
    path_train: str,
    path_dev: str,
    path_test: str,
    save_dir: str,
    num_iter: int,
    with_train: bool,
    by_rel: bool,
    g_encoder_name: str = 'generate',
    diverse: bool = False, 
    score_only_ext: bool = False, 
    limit: int = 5000,
    num_gen_per_label: int = 250, 
):
    print(dict(main_dual_loss=locals()))
    generator = Generator(
        load_dir="gpt2",
        save_dir=str(Path(save_dir) / "generator/iter0"),
        num_gen_per_label=num_gen_per_label, 
        encoder_name=g_encoder_name,
    )
    extractor = Extractor(
        load_dir="facebook/bart-base",
        save_dir=str(Path(save_dir) / "extractor/iter0"),
    )

    generator.fit(path_train, path_dev)
    extractor.fit(path_train, path_dev)
    
    labels_dev = Dataset.load(path_dev).get_labels()
    labels_test = Dataset.load(path_test).get_labels()
    for i in range(num_iter):
        path_synthetic = str(Path(save_dir) / "synthetic" / f"{i}.jsonl")
        path_synthetic_generator = str(Path(save_dir) / "synthetic" / f"{i}_gen.jsonl")
        path_synthetic_extractor = str(Path(save_dir) / "synthetic" / f"{i}_ext.jsonl")
        generator.generate(labels_dev + labels_test, path_out=path_synthetic)
        if not score_only_ext:
            generator.estimate(path_synthetic, path_synthetic)
        extractor.estimate(path_synthetic, path_synthetic)
        

        # filter
        path_filtered = str(Path(save_dir) / "filtered" / f"{i}.jsonl")
        filter_data(path_synthetic, path_train, path_filtered, generator.num_gen_per_label / 2, (i + 1.) / num_iter, with_train, by_rel, model_name=str(Path(save_dir) / "generator" / f'iter{i}' / "model"), diverse=diverse)
        # filter_data(path_synthetic, path_train, path_filtered, generator.num_gen_per_label / 2, (i + 1.) / num_iter, with_train, by_rel, model_name=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"), diverse=diverse)


        extractor = Extractor(
            load_dir=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "extractor" / f'iter{i+1}'),
        )
        generator = Generator(
            load_dir=str(Path(save_dir) / "generator" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "generator" / f'iter{i+1}'),
            num_gen_per_label=num_gen_per_label, 
            encoder_name=g_encoder_name,
        )
        extractor.fit(path_filtered, path_dev)
        generator.fit(path_filtered, path_dev)
        run_eval(path_model=str(Path(save_dir) / "extractor" / f'iter{i+1}'), 
                 path_test=path_dev, mode='single', is_eval=True, limit=0)
    eval_best(path_test=path_test, save_dir=save_dir, num_iter=num_iter, limit=0)




def main_many(data_dir_pattern: str, save_dir: str, **kwargs):
    mode = Path(save_dir).name
    assert mode in ["fewrel", "wiki"]
    records = []

    for path in tqdm(sorted(Path().glob(data_dir_pattern))):
        path_train = path / "train.jsonl"
        path_dev = path / "dev.jsonl"
        path_test = path / "test.jsonl"
        results = main(
            path_train=str(path_train),
            path_dev=str(path_dev),
            path_test=str(path_test),
            save_dir=str(Path(save_dir) / path.name),
            **kwargs,
        )
        records.append(results)

    avg_p = sum([r["precision"] for r in records]) / len(records)
    avg_r = sum([r["recall"] for r in records]) / len(records)
    avg_f = safe_divide(2 * avg_p * avg_r, avg_p + avg_r)
    info = dict(avg_p=avg_p, avg_r=avg_r, avg_f=avg_f)
    print(json.dumps(info, indent=2))

def eval_best(path_test: str, save_dir: str, num_iter: int, limit: int):
    print(dict(eval_best=locals()))
    best = 0.
    best_i = -1
    for i in range(num_iter):
        metric = 0 
        for mode in ['single']:
            path_model = str(Path(save_dir) /'extractor' / f'iter{i+1}')
            is_eval = f'is_eval_True'
            path_results = str(Path(path_model) / f"results_{mode}_{is_eval}.json") if limit == 0 else str(Path(path_model) / f"results_{mode}_{is_eval}_limit{limit}.json")
            with open(path_results) as f:
                data = json.load(f)['score']
                metric += data
        if metric > best:
            best = metric 
            best_i = i
    print('best eval metric is', best, 'at iter', best_i+1)
    path_model = str(Path(save_dir) / 'extractor' / f'iter{best_i+1}')
    run_eval(path_model=path_model, path_test=path_test, mode='single', is_eval=False)
    run_eval(path_model=path_model, path_test=path_test, mode='multi', is_eval=False)
    # run_eval(path_model=path_model, path_test=path_test, mode='all_single', is_eval=False)
    # run_eval(path_model=path_model, path_test=path_test, mode='all_multi', is_eval=False)


def run_eval(path_model: str, path_test: str, mode: str, is_eval: bool, limit: int = 0):
    print(dict(run_eval=locals()))
    is_eval = f'is_eval_{is_eval}'
    path_results = str(Path(path_model) / f"results_{mode}_{is_eval}.json") if limit == 0 else str(Path(path_model) / f"results_{mode}_{is_eval}_limit{limit}.json")
    if Path(path_results).exists():
        return 
    data = Dataset.load(path_test)
    model = Extractor(load_dir=str(Path(path_model) / "model"), save_dir=path_model)

    if mode == "single":
        data.sents = [s for s in data.sents if len(s.triplets) == 1]
    elif mode == "multi":
        data.sents = [s for s in data.sents if len(s.triplets) > 1]
    elif 'all' in mode:
        pass
    else:
        raise ValueError(f"mode must be single or multi")

    if limit > 0:
        random.seed(0)
        random.shuffle(data.sents)
        data.sents = data.sents[:limit]
    path_in = str(Path(path_model) / f"pred_in_{mode}_{is_eval}.jsonl") if limit == 0 else str(Path(path_model) / f"pred_in_{mode}_{is_eval}_limit{limit}.jsonl")
    path_out = str(Path(path_model) / f"pred_out_{mode}_{is_eval}.jsonl") if limit == 0 else str(Path(path_model) / f"pred_out_{mode}_{is_eval}_limit{limit}.jsonl")
    data.save(path_in)

    if "single" in mode:
        model.predict(path_in, path_out)
    else:
        model.predict_multi(path_in, path_out)

    results = model.score(path_pred=path_out, path_gold=path_in)
    
    results.update(mode=mode, limit=limit, path_results=path_results)
    print(json.dumps(results, indent=2))
    with open(path_results, "w") as f:
        json.dump(results, f, indent=2)


def run_eval_many(path_model_pattern: str, data_dir: str, **kwargs):
    for path in tqdm(sorted(Path().glob(path_model_pattern))):
        name = path.parts[-2]
        path_test = Path(data_dir) / name / "test.jsonl"
        assert path_test.exists()
        run_eval(path_model=str(path), path_test=str(path_test), **kwargs)


"""
FewRel Dataset

python wrapper.py main \
--path_train outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/train.jsonl \
--path_dev outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/dev.jsonl \
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--save_dir outputs/wrapper/fewrel/unseen_10_seed_0

python wrapper.py run_eval \
--path_model outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--mode single

python wrapper.py run_eval \
--path_model outputs/wrapper/fewrel/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/fewrel/unseen_10_seed_0/test.jsonl \
--mode multi

Wiki-ZSL Dataset

python wrapper.py main \
--path_train outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/train.jsonl \
--path_dev outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/dev.jsonl \
--path_test outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/test.jsonl \
--save_dir outputs/wrapper/wiki/unseen_10_seed_0

python wrapper.py run_eval \
--path_model outputs/wrapper/wiki/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/test.jsonl \
--mode single

python wrapper.py run_eval \
--path_model outputs/wrapper/wiki/unseen_10_seed_0/extractor_final \
--path_test outputs/data/splits/zero_rte/wiki/unseen_10_seed_0/test.jsonl \
--mode multi

"""


if __name__ == "__main__":
    Fire()
