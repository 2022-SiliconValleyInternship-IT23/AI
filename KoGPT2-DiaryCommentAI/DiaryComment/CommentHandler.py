import torch
import numpy
from transformers import PreTrainedTokenizerFast
from ts.torch_handler.base_handler import BaseHandler

class CommentHandler(BaseHandler):
  def inference(self, data):
    tok = TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                              bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                              pad_token='<pad>', mask_token='<unused0>')
    with torch.no_grad():
      while 1:
        user = data[0].strip()
        comment = ''
        while 1:
          input_ids = torch.LongTensor(tok.encode('<usr>' + user + '</s>' + '0' + '<sys>')).unsqueeze(dim=0)
          pred = self.model.forward(input_ids)
          gen = tok.convert_ids_to_tokens(
            torch.argmax(
              pred,
              dim=-1).squeeze().numpy().tolist())[-1]
          if gen == '</s>':
            break
          comment += gen.replace('▁', ' ')
    return comment.strip()
  def postprocess(self, comment):
    res = []
    res.append({'comment': comment})
    return res

    
  