# t5_infer.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5SmallInfer:
    def __init__(self, model_name="t5-small", device=None):
        """
        init the model
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        text,
        max_length=250,
        num_beams=4,
        do_sample=False,
        temperature=1.0
    ):
        """
        do inference and generate text
        Args:
            text (str): the input text
            max_length (int): the maximum length of the generated text
            num_beams (int): the number of beams for beam search
            do_sample (bool): whether to use sampling or greedy decoding
            temperature (float): the temperature for sampling
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature
            )

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return result


# # 如果直接运行这个文件，可测试
# if __name__ == "__main__":
#     model = T5SmallInfer()
#     text = "translate English to German: The house is wonderful."
#     output = model.generate(text)
#     print(output)