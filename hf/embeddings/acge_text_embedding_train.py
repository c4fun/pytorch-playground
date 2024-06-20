import torch
import argparse
import functools
from C_MTEB.tasks import *
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from mteb import MTEB, DRESModel


class RetrievalModel(DRESModel):
    def __init__(self, encoder, **kwargs):
        self.encoder = encoder

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ['{}'.format(q) for q in queries]
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        input_texts = ['{}'.format(t) for t in input_texts]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        return self.encoder.encode(
            sentences=input_texts,
            batch_size=512,
            normalize_embeddings=True,
            convert_to_numpy=True
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="acge_text_embedding", type=str)
    parser.add_argument('--task_type', default=None, type=str)
    parser.add_argument('--pooling_method', default='cls', type=str)
    parser.add_argument('--output_dir', default='zh_results',
                        type=str, help='output directory')
    parser.add_argument('--max_len', default=1024, type=int, help='max length')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    encoder = SentenceTransformer(args.model_name_or_path).half()
    encoder.encode = functools.partial(encoder.encode, normalize_embeddings=True)
    encoder.max_seq_length = int(args.max_len)

    task_names = [t.description["name"] for t in MTEB(task_types=args.task_type,
                                                      task_langs=['zh', 'zh-CN']).tasks]
    TASKS_WITH_PROMPTS = ["T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval",
                          "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval"]
    for task in task_names:
        evaluation = MTEB(tasks=[task], task_langs=['zh', 'zh-CN'])
        if task in TASKS_WITH_PROMPTS:
            evaluation.run(RetrievalModel(encoder), output_folder=args.output_dir, overwrite_results=False)
        else:
            evaluation.run(encoder, output_folder=args.output_dir, overwrite_results=False)

