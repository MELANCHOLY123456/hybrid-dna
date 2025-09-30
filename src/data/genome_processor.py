import pyfaidx
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GenomeProcessor:
    """Genome processor for extracting DNA sequences from reference genome.
    
    Supports automatic chromosome name normalization, handles different
    naming conventions, and provides sequence extraction and reverse
    complement functionality.
    """

    def __init__(self, fasta_path):
        """Initialize genome processor.
        
        Args:
            fasta_path: Reference genome FASTA file path
        """
        self.genome = pyfaidx.Fasta(fasta_path)
        self.chromosome_mapping = self._build_chromosome_mapping()
        logger.info(f"Loaded reference genome from {fasta_path}")
        logger.info(f"Available chromosomes: {list(self.genome.keys())[:10]}{'...' if len(self.genome) > 10 else ''}")

    def _build_chromosome_mapping(self) -> dict:
        """Build chromosome name mapping supporting multiple naming conventions.
        
        Creates mapping dictionary from different chromosome name formats to
        actual names in reference genome. Supports with/without 'chr' prefix
        and different mitochondrial chromosome representations.
        
        Returns:
            Chromosome name mapping dictionary
        """
        mapping = {}
        available_chroms = list(self.genome.keys())

        # Create possible name variants for each chromosome
        for chrom in available_chroms:
            # Original name
            mapping[chrom] = chrom

            # Add/remove 'chr' prefix variants
            if chrom.startswith('chr'):
                without_chr = chrom[3:]
                mapping[without_chr] = chrom
            else:
                with_chr = f'chr{chrom}'
                mapping[with_chr] = chrom

            # Handle special chromosome names (mitochondrial)
            if chrom in ['MT', 'chrMT']:
                mapping['M'] = chrom
                mapping['chrM'] = chrom
            elif chrom in ['M', 'chrM']:
                mapping['MT'] = chrom
                mapping['chrMT'] = chrom

        return mapping

    def normalize_chromosome_name(self, chrom: str) -> Optional[str]:
        """Normalize chromosome name to reference genome format.
        
        Tries multiple strategies to match chromosome names:
        1. Direct lookup
        2. Case variants
        3. Add/remove 'chr' prefix
        4. Special chromosome name handling (mitochondrial)
        
        Args:
            chrom: Input chromosome name
            
        Returns:
            Normalized chromosome name, None if no match found
        """
        # Try direct lookup
        if chrom in self.chromosome_mapping:
            return self.chromosome_mapping[chrom]

        # Try case variants
        chrom_lower = chrom.lower()
        for available_chrom in self.genome.keys():
            if available_chrom.lower() == chrom_lower:
                return available_chrom

        # Try add/remove 'chr' prefix
        if chrom.startswith('chr'):
            without_chr = chrom[3:]
            if without_chr in self.genome.keys():
                return without_chr
        else:
            with_chr = f'chr{chrom}'
            if with_chr in self.genome.keys():
                return with_chr

        # Handle special chromosome names (mitochondrial)
        if chrom in ['M', 'MT']:
            for mt_name in ['chrM', 'chrMT', 'M', 'MT']:
                if mt_name in self.genome.keys():
                    return mt_name

        logger.warning(f"Cannot find match for chromosome {chrom}")
        return None

    def get_sequence(self, chrom: str, start: int, end: int, strand: str = '+') -> Optional[str]:
        """Get DNA sequence from specified region with chromosome name handling.
        
        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position (0-based)
            strand: Strand direction ('+' or '-'), default '+'
            
        Returns:
            DNA sequence, None if error occurs
        """
        # Normalize chromosome name
        normalized_chrom = self.normalize_chromosome_name(chrom)
        if normalized_chrom is None:
            logger.error(f"Cannot normalize chromosome name: {chrom}")
            return None

        try:
            # Get sequence
            sequence = str(self.genome[normalized_chrom][start:end])

            # Handle negative strand (get reverse complement)
            if strand == '-':
                sequence = self.reverse_complement(sequence)

            return sequence.upper()

        except Exception as e:
            logger.error(f"Error extracting sequence from {normalized_chrom}:{start}-{end}: {e}")
            return None

    def get_centered_sequence(self, chrom: str, pos: int, window_size: int = 1024) -> Optional[str]:
        """Get DNA sequence centered at specified position.
        
        Args:
            chrom: Chromosome name
            pos: Center position (1-based)
            window_size: Window size, default 1024
            
        Returns:
            DNA sequence centered at position, None if error occurs
        """
        # Normalize chromosome name
        normalized_chrom = self.normalize_chromosome_name(chrom)
        if normalized_chrom is None:
            return None

        # Calculate sequence range
        start = max(0, pos - window_size)

        # Get chromosome length
        try:
            chrom_length = len(self.genome[normalized_chrom])
            # Calculate end position ensuring sequence length is 2 * window_size + 1
            end = min(chrom_length, pos + window_size)
        except KeyError:
            logger.error(f"Cannot get length of chromosome {normalized_chrom}")
            return None

        return self.get_sequence(normalized_chrom, start, end)

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Get reverse complement of DNA sequence.
        
        Args:
            sequence: Input DNA sequence
            
        Returns:
            Reverse complement sequence
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement[base] for base in reversed(sequence))