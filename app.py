import unittest
import math
from enum import Enum

# ==========================================
# CORE ALGORITHM (V5)
# ==========================================
class WindowScale(Enum):
    LINEAR = "linear"
    LOGARITHMIC = "log"
    FIBONACCI = "fib"

class VectorStructure(Enum):
    BY_DIRECTION = "by_direction"
    BY_DISTANCE = "by_distance"

class LevelOfDetail(Enum):
    EXACT = "exact"
    ZONE = "zone"
    CUMULATIVE = "cumulative"

class ContextualEmbedderV5:
    def __init__(self, alphabet_str, max_steps=3, scale=WindowScale.LINEAR, 
                 structure=VectorStructure.BY_DISTANCE, detail=LevelOfDetail.EXACT):
        self.max_steps = max_steps
        self.scale = scale
        self.structure = structure
        self.detail = detail
        
        special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
        self.alphabet_map = {char: i for i, char in enumerate(alphabet_str)}
        next_dim = len(self.alphabet_map)
        for token in special_tokens:
            self.alphabet_map[token] = next_dim
            next_dim += 1
            
        self.num_dims = len(self.alphabet_map)
        self.distances = self._generate_distances()
        
    def _generate_distances(self):
        if self.scale == WindowScale.LINEAR:
            return [i for i in range(1, self.max_steps + 1)]
        elif self.scale == WindowScale.LOGARITHMIC:
            return [2**i for i in range(self.max_steps)]
        elif self.scale == WindowScale.FIBONACCI:
            fib = [1, 2]
            while len(fib) < self.max_steps:
                fib.append(fib[-1] + fib[-2])
            return fib[:self.max_steps]
        return [i for i in range(1, self.max_steps + 1)]
        
    def _get_dim(self, char):
        return self.alphabet_map.get(char, self.alphabet_map['<UNK>'])

    def fit_transform(self, text):
        padded_text = ['<SOS>'] + list(text) + ['<EOS>']
        embedding_matrix = []
        for i in range(1, len(padded_text) - 1):
            sparse_emb = self._extract_recursive(padded_text, i)
            dense_vec = self._to_dense(sparse_emb)
            embedding_matrix.append(dense_vec)
        return embedding_matrix

    def _extract_recursive(self, padded_text, target_idx):
        emb = {
            'center': self._get_dim(padded_text[target_idx]),
            'left': {},
            'right': {}
        }
        
        for step in range(self.max_steps):
            emb['left'][step] = [0.0] * self.num_dims
            emb['right'][step] = [0.0] * self.num_dims
            
        prev_dist = 0
        
        for step, dist in enumerate(self.distances):
            if self.detail == LevelOfDetail.EXACT:
                start_d = dist
                end_d = dist
            elif self.detail == LevelOfDetail.ZONE:
                start_d = prev_dist + 1
                end_d = dist
            elif self.detail == LevelOfDetail.CUMULATIVE:
                start_d = 1
                end_d = dist
            else:
                start_d = dist
                end_d = dist
                
            for d in range(start_d, end_d + 1):
                left_idx = target_idx - d
                if left_idx >= 0:
                    dim = self._get_dim(padded_text[left_idx])
                    emb['left'][step][dim] += 1.0
                    
                right_idx = target_idx + d
                if right_idx < len(padded_text):
                    dim = self._get_dim(padded_text[right_idx])
                    emb['right'][step][dim] += 1.0
                    
            prev_dist = dist
            
        return emb

    def _to_dense(self, sparse_emb):
        vec = []
        center_one_hot = [0.0] * self.num_dims
        center_one_hot[sparse_emb['center']] = 1.0
        vec.extend(center_one_hot)
        
        if self.structure == VectorStructure.BY_DIRECTION:
            for step in range(self.max_steps):
                vec.extend(sparse_emb['left'][step])
            for step in range(self.max_steps):
                vec.extend(sparse_emb['right'][step])
        else:
            for step in range(self.max_steps):
                vec.extend(sparse_emb['left'][step])
                vec.extend(sparse_emb['right'][step])
            
        return vec


# ==========================================
# COMPREHENSIVE TEST SUITE
# ==========================================

class TestContextualEmbedderV5(unittest.TestCase):
    
    def setUp(self):
        self.alphabet = "ABCDTHE "
        
    def test_1_distance_generation(self):
        """Verify that mathematical scales generate the correct distances."""
        emb_linear = ContextualEmbedderV5(self.alphabet, max_steps=3, scale=WindowScale.LINEAR)
        self.assertEqual(emb_linear.distances, [1, 2, 3])
        
        emb_log = ContextualEmbedderV5(self.alphabet, max_steps=3, scale=WindowScale.LOGARITHMIC)
        self.assertEqual(emb_log.distances, [1, 2, 4])
        
        emb_fib = ContextualEmbedderV5(self.alphabet, max_steps=4, scale=WindowScale.FIBONACCI)
        self.assertEqual(emb_fib.distances, [1, 2, 3, 5])

    def test_2_vector_structure_layout(self):
        """Verify that BY_DIRECTION and BY_DISTANCE alter the 1D array sequence correctly."""
        text = "A"
        emb_dir = ContextualEmbedderV5(self.alphabet, max_steps=2, structure=VectorStructure.BY_DIRECTION)
        emb_dist = ContextualEmbedderV5(self.alphabet, max_steps=2, structure=VectorStructure.BY_DISTANCE)
        
        vec_dir = emb_dir.fit_transform(text)[0]
        vec_dist = emb_dist.fit_transform(text)[0]
        
        # Total length should be identical: 1 Center + 2 Left + 2 Right = 5 blocks
        expected_len = 5 * emb_dir.num_dims
        self.assertEqual(len(vec_dir), expected_len)
        self.assertEqual(len(vec_dist), expected_len)
        
        # In BY_DISTANCE, Left Step 1 starts at index 1 * num_dims
        # In BY_DIRECTION, Left Step 1 also starts at index 1 * num_dims
        # For text "A", Left Step 1 (dist 1) is <SOS>
        dim_sos = emb_dir._get_dim('<SOS>')
        self.assertEqual(vec_dir[emb_dir.num_dims + dim_sos], 1.0)
        self.assertEqual(vec_dist[emb_dist.num_dims + dim_sos], 1.0)
        
        # In BY_DIRECTION, Right Step 1 starts after Center + Left1 + Left2 (3 * num_dims)
        # In BY_DISTANCE, Right Step 1 starts after Center + Left1 (2 * num_dims)
        # In text "A", with max_steps=2, Right 1 is <EOS>, Right 2 is <PAD>
        dim_eos = emb_dir._get_dim('<EOS>')
        self.assertEqual(vec_dir[3 * emb_dir.num_dims + dim_eos], 1.0)
        self.assertEqual(vec_dist[2 * emb_dist.num_dims + dim_eos], 1.0)

    def test_3_sequence_preservation(self):
        """Verify that 'THE' and 'HET' produce different vectors for 'E'."""
        emb = ContextualEmbedderV5(self.alphabet, max_steps=2, detail=LevelOfDetail.EXACT)
        
        matrix_the = emb.fit_transform("THE")
        matrix_het = emb.fit_transform("HET")
        
        # 'E' is at index 2 in "THE" and index 1 in "HET"
        vec_E_the = matrix_the[2]
        vec_E_het = matrix_het[1]
        
        # Vectors must not be identical
        self.assertNotEqual(vec_E_the, vec_E_het)
        
        # In "THE", 'T' should be at Left Step 2
        dim_T = emb._get_dim('T')
        left_2_start_idx = (1 + 2) * emb.num_dims # Center + L1 + L2 offset
        self.assertEqual(vec_E_the[left_2_start_idx + dim_T], 1.0)
        
        # In "HET", 'T' should be at Right Step 1
        right_1_start_idx = (1 + 1) * emb.num_dims # Center + L1 + R1 offset (BY_DISTANCE)
        self.assertEqual(vec_E_het[right_1_start_idx + dim_T], 1.0)

    def test_4_level_of_detail_exact(self):
        """Verify EXACT detail only looks at the precise distance."""
        text = "ABCDE"
        # target 'C' is index 2. Distances [1, 2]
        emb = ContextualEmbedderV5(self.alphabet, max_steps=2, scale=WindowScale.LINEAR, detail=LevelOfDetail.EXACT)
        matrix = emb.fit_transform(text)
        vec_C = matrix[2]
        
        dim_B = emb._get_dim('B')
        dim_A = emb._get_dim('A')
        
        # Step 0 (Dist 1) should catch 'B' (Left 1)
        left_1_start = (1 + 0) * emb.num_dims
        self.assertEqual(vec_C[left_1_start + dim_B], 1.0)
        
        # Step 1 (Dist 2) should catch 'A' (Left 2)
        left_2_start = (1 + 2) * emb.num_dims
        self.assertEqual(vec_C[left_2_start + dim_A], 1.0)

    def test_5_level_of_detail_zone(self):
        """Verify ZONE detail looks at non-overlapping blocks."""
        text = "ABCDE"
        # target 'C' is index 2. Log Distances: [1, 2, 4]
        emb = ContextualEmbedderV5(self.alphabet, max_steps=3, scale=WindowScale.LOGARITHMIC, detail=LevelOfDetail.ZONE)
        matrix = emb.fit_transform(text)
        vec_C = matrix[2]
        
        dim_B = emb._get_dim('B')
        dim_A = emb._get_dim('A')
        dim_SOS = emb._get_dim('<SOS>')
        
        # Step 0 (Dist 1, Zone [1]) should catch 'B' (Left 1)
        left_1_start = (1 + 0) * emb.num_dims
        self.assertEqual(vec_C[left_1_start + dim_B], 1.0)
        
        # Step 1 (Dist 2, Zone [2]) should catch 'A' (Left 2)
        left_2_start = (1 + 2) * emb.num_dims
        self.assertEqual(vec_C[left_2_start + dim_A], 1.0)
        
        # Step 2 (Dist 4, Zone [3, 4]) should catch '<SOS>' (Left 3 & 4 are out of bounds)
        # Left 3 is out of bounds (idx -1), Left 4 is out of bounds (idx -2)
        left_3_start = (1 + 4) * emb.num_dims
        self.assertEqual(vec_C[left_3_start + dim_SOS], 1.0)

    def test_6_level_of_detail_cumulative(self):
        """Verify CUMULATIVE detail builds an expanding overlapping histogram."""
        text = "ABCDE"
        # target 'C' is index 2. Distances [1, 2]
        emb = ContextualEmbedderV5(self.alphabet, max_steps=2, scale=WindowScale.LINEAR, detail=LevelOfDetail.CUMULATIVE)
        matrix = emb.fit_transform(text)
        vec_C = matrix[2]
        
        dim_B = emb._get_dim('B')
        dim_A = emb._get_dim('A')
        
        # Step 0 (Dist 1, Cum [1]) should catch 'B' (count=1)
        left_1_start = (1 + 0) * emb.num_dims
        self.assertEqual(vec_C[left_1_start + dim_B], 1.0)
        
        # Step 1 (Dist 2, Cum [1, 2]) should catch 'B' (count=1) AND 'A' (count=1)
        left_2_start = (1 + 2) * emb.num_dims
        self.assertEqual(vec_C[left_2_start + dim_B], 1.0) # B is still counted
        self.assertEqual(vec_C[left_2_start + dim_A], 1.0) # A is now counted

    def test_7_unicode_support(self):
        """Verify the algorithm handles Unicode alphabets and characters flawlessly."""
        uni_alphabet = "ABC🚀🐶"
        emb = ContextualEmbedderV5(uni_alphabet, max_steps=1, detail=LevelOfDetail.EXACT)
        
        text = "A🚀C"
        matrix = emb.fit_transform(text)
        
        # Target '🚀' (index 1)
        vec_rocket = matrix[1]
        
        dim_A = emb._get_dim('A')
        dim_C = emb._get_dim('C')
        dim_rocket = emb._get_dim('🚀')
        
        # Center is Rocket
        self.assertEqual(vec_rocket[dim_rocket], 1.0)
        
        # Left 1 is A
        left_1_start = (1 + 0) * emb.num_dims
        self.assertEqual(vec_rocket[left_1_start + dim_A], 1.0)
        
        # Right 1 is C
        # In BY_DISTANCE, Right 1 is at block 2 (Center + L1)
        right_1_start = (1 + 1) * emb.num_dims
        self.assertEqual(vec_rocket[right_1_start + dim_C], 1.0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)