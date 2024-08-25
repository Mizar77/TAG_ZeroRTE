from pydantic.main import BaseModel
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from models import JointModel, Config
from data import JointTrainer
import json 
import os 
import torch 

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_class: Optional[str] = field(default='JointModel')
    model_read_ckpt: Optional[str] = field(default=None,)
    model_write_ckpt: Optional[str] = field(default=None,)
    pretrained_wv: Optional[str] = field(default='./wv/glove.6B.100d.txt', )
    # dataset: Optional[str] = field(default='ACE05')
    label_config: Optional[str] = field(default=None)
    batch_size: Optional[int] = field(default=24)
    evaluate_interval: Optional[int] = field(default=500,)
    max_steps: Optional[int] = field(default=4000,)
    max_epoches: Optional[int] = field(default=20,)
    decay_rate: Optional[float] = field(default=0.05)
    #### Model Config
    token_emb_dim: Optional[int] = field(default=100,)
    char_encoder: Optional[str] = field(default='lstm',)
    char_emb_dim: Optional[int] = field(default=30,)
    cased: Optional[int] = field(default=False,)
    hidden_dim: Optional[int] = field(default=200,)
    num_layers: Optional[int] = field(default=3,)
    crf: Optional[str] = field(default=None,)
    loss_reduction: Optional[str] = field(default='sum',)
    maxlen: Optional[int] = field(default=None,)
    dropout: Optional[float] = field(default=0.5,)
    optimizer: Optional[str] = field(default='adam',)
    lr: Optional[float] = field(default=0.001,)
    vocab_size: Optional[int] = field(default=30000)
    vocab_file: Optional[str] = field(default=None)
    ner_tag_vocab_size: Optional[int] = field(default=64)
    re_tag_vocab_size: Optional[int] = field(default=250,)
    lm_emb_dim: Optional[int] = field(default=0)
    lm_emb_path: Optional[str] = field(default='albert-base-v2')
    head_emb_dim: Optional[int] = field(default=0)
    tag_form: Optional[str] = field(default='iob2')
    warm_steps: Optional[int] = field(default=1000,)
    grad_period: Optional[int] = field(default=1,)
    device: Optional[str] = field(default='cuda',)
    seed: Optional[int] = field(default=42,)

class Extractor(BaseModel):
    load_dir: str
    save_dir: str
    epoches: int = 20
    steps: int = 4000
    pretrained: str = 'albert-base-v2'
    search_threshold: float = -0.9906
    model_kwargs: dict = {}

    def get_train_args(self, pretrained_wv=None, vocab_size=None, model_read_ckpt=None, model_write_ckpt=None,) -> ModelArguments:
        def get_dim(x):
            if 'albert-base' in x:
                return 768
            elif 'albert-large' in x:
                return 1024 
            elif 'albert-xlarge' in x:
                return 2048
            elif 'albert-xllarge' in x:
                return 4096 
            else:
                raise ValueError(f'wrong pretrained model {x}')
        def get_head_dim(x):
            if 'albert-base' in x:
                return 144
            elif 'albert-large' in x:
                return 384 
            elif 'albert-xlarge' in x:
                return 768
            elif 'albert-xllarge' in x:
                return 768 
            else:
                raise ValueError(f'wrong pretrained model {x}')
        args = ModelArguments(
                lm_emb_path=self.pretrained,
                lm_emb_dim=get_dim(self.pretrained),
                head_emb_dim=get_head_dim(self.pretrained),
                pretrained_wv=pretrained_wv,
                vocab_size=vocab_size, 
                model_read_ckpt=model_read_ckpt,
                model_write_ckpt=model_write_ckpt,
                max_epoches=self.epoches,
                max_steps=self.steps,
            )
        return args
    def get_wv_path(self, mode):
        return f'{self.save_dir}/data/{mode}.txt'

    def process_wv(self, paths, mode, all_wv_path='wv/glove.6B.100d.txt'):
        if isinstance(paths, str):
            paths = [paths]
        out_path = self.get_wv_path(mode)
        all_tokens = set()
        for path in paths:
            with open(path, encoding='utf-8') as f:
                data = json.loads(f.readline())
                for line in data:
                    tokens = line['tokens']
                    all_tokens.update(set([x.lower() for x in tokens]))
        vocab_size = 0 
        with open(all_wv_path, encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as f_out:
            for line in f:
                token = line.split()[0]
                if token in all_tokens:
                    f_out.write(line)
                    vocab_size += 1
        print(f'{mode} vocab size: {vocab_size}')
        return vocab_size

    def get_model(self, args):
        # args = self.get_train_args(pretrained_wv=self.get_wv_path(mode), vocab_size=vocab_size, model_read_ckpt=model_read_ckpt, model_write_ckpt=model_write_ckpt)
        config = Config(**args.__dict__)
        print(dict(get_model={'config': config.__dict__}))
        model = JointModel(config)
        if args.model_read_ckpt != '':
            model.load_ckpt(args.model_read_ckpt)
        if args.token_emb_dim > 0 and args.pretrained_wv:
            print(f"reading pretrained wv from {args.pretrained_wv}")
            model.token_embedding.token_indexing.update_vocab = True
            model.token_embedding.token_indexing.vocab = {
                '[PAD]': 0,
                '[MASK]': 1,
                '[CLS]': 2,
                '[SEP]': 3,
                '[UNK]': 4,
            }
            _w = model.token_embedding.token_embedding.weight 
            _w_data = _w.data
            _w.data = torch.zeros([args.vocab_size, config.token_emb_dim], dtype=_w.dtype, device=_w.device)
            model.token_embedding.token_indexing.inv_vocab = {}
            model.token_embedding.load_pretrained(args.pretrained_wv, freeze=True)
            _w.data[:5] = _w_data[:5]
            model.token_embedding.token_indexing.update_vocab = False
        return model

    def write_data(self, path_in, path_out):
        with open(path_in, encoding='utf-8') as f, open(path_out, 'w', encoding='utf-8') as f_out:
            data = [json.loads(line) for line in f]
            new_data = []
            for ins in data:
                tokens = ''
                entities = set()
                relations = []
                rewards = []
                for tri in ins['triplets']:
                    if tokens == '':
                        tokens = tri['tokens']
                    else:
                        assert ' '.join(tokens) == ' '.join(tri['tokens'])
                    h, t = tri['head'], tri['tail']
                    if len(h) == 0 or len(t) == 0:
                        continue
                    entities.add((h[0], h[-1] + 1, 'entity'))
                    entities.add((t[0], t[-1] + 1, 'entity'))
                    relations.append((h[0], h[-1] + 1, t[0], t[-1] + 1, tri['label']))
                    if 'reward' in tri:
                        rewards.append(tri['reward'])
                    else:   
                        rewards.append(1.0)
                new_data.append({'tokens': tokens, 'entities': list(entities), 'relations': relations, 'reward': sum(rewards) / len(rewards) if len(rewards) != 0 else 1.0})
            json.dump(new_data, f_out)
    def get_saved_model_dir(self):
        # os.makedirs(f'{self.save_dir}/model', exist_ok=True)
        return f'{self.save_dir}/model'
    
    def get_load_model_dir(self):
        return f'{self.load_dir}'
    
    def get_data_dir(self):
        os.makedirs(f'{self.save_dir}/data', exist_ok=True)
        return f'{self.save_dir}/data'

    def fit(self, path_train: str, path_dev: str):
        print('fit', dict(path_train=path_train, path_dev=path_dev))
        if Path(f'{self.save_dir}/model.json').exists():
            return
        mode='train'
        os.makedirs(self.get_saved_model_dir(), exist_ok=True)
        self.write_data(path_train, f'{self.get_data_dir()}/train.json')
        self.write_data(path_dev, f'{self.get_data_dir()}/dev.json')
        vocab_size = self.process_wv([f'{self.get_data_dir()}/train.json', f'{self.get_data_dir()}/dev.json'], mode)
        args = self.get_train_args(pretrained_wv=self.get_wv_path(mode), vocab_size=vocab_size+100, model_read_ckpt=self.get_load_model_dir(), model_write_ckpt=self.get_saved_model_dir())
        model = self.get_model(args)
        trainer = JointTrainer(model=model,
                    train_path=f'{self.get_data_dir()}/train.json',
                    valid_path=f'{self.get_data_dir()}/dev.json',
                    batch_size=self.get_train_args().batch_size, 
                    tag_form=self.get_train_args().tag_form,
                    num_workers=0,)
        trainer.train_model(args=args)

    def estimate(self, path_in, path_out):
        print(dict(estimate={'path_in': path_in, 'path_out': path_out}))
        # if Path(path_out).exists():
        #     return 
        mode='estimate'
        path_mid = f'{self.get_data_dir()}/estimate.json'
        self.write_data(path_in, path_mid)
        vocab_size = self.process_wv(path_mid, mode)
        args = self.get_train_args(pretrained_wv=self.get_wv_path(mode), vocab_size=vocab_size+100,model_read_ckpt=self.get_saved_model_dir())
        model = self.get_model(args)
        trainer = JointTrainer(model=model)
        dataloader = trainer.get_dataloader(path_mid, shuffle=False, batch_size=args.batch_size, tag_form=args.tag_form)
        data = trainer.estimate(model, dataloader)
        with open(path_out, 'w', encoding='utf-8') as f:
            for ins in data:
                ret = []
                tokens = ins['tokens']
                for pred in ins['pred']:
                    h0, h1, t0, t1, r = pred['rel']
                    nll = pred['nll']
                    ret.append({'tokens': tokens, 'head': [i for i in range(h0, h1)], 'tail': [i for i in range(t0, t1)], 'label': r, "extractor_nll": nll})
                f.write(json.dumps({'triplets': ret}) + '\n')
    
    def predict(self, path_in, path_out, labels):
        print(dict(predict={'path_in': path_in, 'path_out': path_out}))
        if Path(path_out).exists():
            return 
        mode='predict'
        path_mid = f'{self.get_data_dir()}/predict.json'
        self.write_data(path_in, path_mid)
        vocab_size = self.process_wv(path_mid, mode)
        args = self.get_train_args(pretrained_wv=self.get_wv_path(mode), vocab_size=vocab_size+100,model_read_ckpt=self.get_saved_model_dir())
        model = self.get_model(args)
        trainer = JointTrainer(model=model)
        dataloader = trainer.get_dataloader(path_mid, shuffle=False, batch_size=args.batch_size, tag_form=args.tag_form)
        data = trainer.predict(model, dataloader, labels)
        with open(path_out, 'w', encoding='utf-8') as f:
            for ins in data:
                f.write(json.dumps(ins) + '\n')


