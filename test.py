import kenlm
import numpy as np

# Load your n-gram LM
lm = kenlm.Model("models/checkpoints/kenlm.binary")

def beam_search_with_lm(ctc_probs, vocab, beam_width=5, alpha=0.5):
    """
    ctc_probs: numpy array of shape (T, vocab_size) - CTC probabilities per frame
    vocab: list of glosses (e.g., ["HELLO", "I", "HAVE", ...])
    beam_width: number of hypotheses to keep
    alpha: weight for LM score
    """
    T, V = ctc_probs.shape
    beam = [("", 0.0)]  # (hypothesis, score)

    for t in range(T):
        candidates = {}
        for hyp, score in beam:
            # Extend with blank (CTC's blank token, often index 0)
            candidates[hyp] = candidates.get(hyp, -float('inf')) + np.log(ctc_probs[t, 0])

            # Extend with each gloss
            for i, gloss in enumerate(vocab):
                if i == 0: continue  # Skip blank
                new_hyp = (hyp + " " + gloss).strip()
                ctc_score = score + np.log(ctc_probs[t, i])
                lm_score = lm.score(new_hyp)
                total_score = ctc_score + alpha * lm_score
                candidates[new_hyp] = max(candidates.get(new_hyp, -float('inf')), total_score)

        # Keep top beam_width hypotheses
        beam = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:beam_width]
    
    return beam[0][0]  # Return best sequence

# Example usage
vocab = ["", "HELLO", "I", "HAVE", "GOOD", "LUNCH", "YOU", "AND", "WHAT", "ARE", "DO"]  # "" is blank
ctc_probs = np.random.rand(20, len(vocab))  # Dummy CTC output (20 frames)
ctc_probs = ctc_probs / ctc_probs.sum(axis=1, keepdims=True)  # Normalize to probabilities
result = beam_search_with_lm(ctc_probs, vocab)
print(f"Predicted sequence: {result}")