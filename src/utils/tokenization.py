class DNATokenizer:
    """DNA sequence tokenizer.
    
    Converts DNA base sequences to integer sequences for neural network processing.
    Supports encoding (sequence->integers) and decoding (integers->sequence) operations.
    """

    def __init__(self):
        """Initialize DNA tokenizer.
        
        Defines base-to-integer mapping:
        - A: 0 (Adenine)
        - C: 1 (Cytosine)
        - G: 2 (Guanine)
        - T: 3 (Thymine)
        - N: 4 (Unknown base)
        """
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, sequence):
        """Encode DNA sequence to integer sequence.
        
        Args:
            sequence: DNA base sequence, e.g. "ACGT"
            
        Returns:
            Integer sequence, e.g. [0, 1, 2, 3]
        """
        return [self.vocab.get(base, 4) for base in sequence.upper()]

    def decode(self, tokens):
        """Decode integer sequence to DNA sequence.
        
        Args:
            tokens: Integer sequence, e.g. [0, 1, 2, 3]
            
        Returns:
            DNA base sequence, e.g. "ACGT"
        """
        return ''.join([self.inverse_vocab.get(token, 'N') for token in tokens])

    @property
    def vocab_size(self):
        """Get vocabulary size.
        
        Returns:
            Vocabulary size including 5 bases (A,C,G,T,N)
        """
        return len(self.vocab)