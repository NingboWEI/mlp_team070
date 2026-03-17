# all the inference code for every model goes here
# every model is a class with a generate() method that takes in a string and returns a string

class  Test_model:
    def __init__(self, device=None):
        """
        init the model
        """

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
        result = text

        return result