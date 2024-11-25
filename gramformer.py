# gramformer.py
class Gramformer:
    def __init__(self, models=1, use_gpu=False):
        from transformers import AutoTokenizer
        from transformers import AutoModelForSeq2SeqLM
        import errant
        import re
        self.annotator = errant.load('en')
        self.re = re
        
        if use_gpu:
            device = "cuda:0"
        else:
            device = "cpu"
            
        self.device = device
        correction_model_tag = "prithivida/grammar_error_correcter_v1"
        self.model_loaded = False

        if models == 1:
            self.correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
            self.correction_model = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)
            self.correction_model = self.correction_model.to(device)
            self.model_loaded = True
            print("[Gramformer] Grammar error correct/highlight model loaded..")

    def _fix_capitalization(self, text):
        """Fix capitalization issues"""
        # Capitalize first letter of each sentence
        sentences = self.re.split('([.!?]+)', text)
        for i in range(0, len(sentences)-1, 2):
            if sentences[i]:
                sentences[i] = sentences[i].strip()
                if sentences[i]:
                    sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        
        text = ''.join(sentences)
        
        # Capitalize specific words
        always_capitalize = ['i', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                           'saturday', 'sunday', 'january', 'february', 'march', 'april', 
                           'may', 'june', 'july', 'august', 'september', 'october', 
                           'november', 'december']
        
        for word in always_capitalize:
            text = self.re.sub(rf'\b{word}\b', word.capitalize(), text, flags=self.re.IGNORECASE)
        
        return text

    def _fix_punctuation(self, text):
        """Fix common punctuation issues"""
        # Fix redundant words
        text = self.re.sub(r'\b(never)\b[^.!?]*\b\1\b', r'\1', text, flags=self.re.IGNORECASE)
        text = self.re.sub(r'\b(always)\b[^.!?]*\b\1\b', r'\1', text, flags=self.re.IGNORECASE)
        
        # Fix spaces and punctuation
        text = self.re.sub(r'\s+', ' ', text).strip()
        text = self.re.sub(r'\s+([.,!?])', r'\1', text)
        text = self.re.sub(r'([.,!?]){2,}', r'\1', text)
        text = self.re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
        
        # Fix common compounds
        text = self.re.sub(r'\balot\b', 'a lot', text, flags=self.re.IGNORECASE)
        text = self.re.sub(r'\batleast\b', 'at least', text, flags=self.re.IGNORECASE)
        text = self.re.sub(r'\beachother\b', 'each other', text, flags=self.re.IGNORECASE)
        
        return text

    def _fix_contractions(self, text):
        """Fix missing apostrophes in contractions"""
        contractions = {
            r'\b(dont)\b': "don't",
            r'\b(cant)\b': "can't",
            r'\b(wont)\b': "won't",
            r'\b(didnt)\b': "didn't",
            r'\b(doesnt)\b': "doesn't",
            r'\b(wouldnt)\b': "wouldn't",
            r'\b(couldnt)\b': "couldn't",
            r'\b(shouldnt)\b': "shouldn't",
            r'\b(hasnt)\b': "hasn't",
            r'\b(havent)\b': "haven't",
            r'\b(hadnt)\b': "hadn't",
            r'\b(isnt)\b': "isn't",
            r'\b(arent)\b': "aren't",
            r'\b(wasnt)\b': "wasn't",
            r'\b(werent)\b': "weren't"
        }
        
        for pattern, replacement in contractions.items():
            text = self.re.sub(pattern, replacement, text, flags=self.re.IGNORECASE)
        return text

    def _fix_grammar(self, text):
        """Fix common grammar patterns"""
        # Fix subject-verb agreement
        text = self.re.sub(r'\b(they|we|you|I)\s+is\b', r'\1 are', text, flags=self.re.IGNORECASE)
        text = self.re.sub(r'\b(he|she|it)\s+are\b', r'\1 is', text, flags=self.re.IGNORECASE)
        
        # Fix article usage
        text = self.re.sub(r'\ba\s+([aeiouAEIOU][a-zA-Z]+)', r'an \1', text)
        
        return text

    def correct(self, input_sentence, max_candidates=1):
        if self.model_loaded:
            # Pre-process
            input_sentence = self._fix_grammar(input_sentence)
            input_sentence = self._fix_contractions(input_sentence)
            
            # Get model correction
            correction_prefix = "gec: "
            input_sentence = correction_prefix + input_sentence
            input_ids = self.correction_tokenizer.encode(input_sentence, return_tensors='pt')
            input_ids = input_ids.to(self.device)

            preds = self.correction_model.generate(
                input_ids,
                do_sample=True, 
                max_length=128, 
                num_beams=7,
                early_stopping=True,
                num_return_sequences=max_candidates)

            corrected = set()
            for pred in preds:  
                correction = self.correction_tokenizer.decode(pred, skip_special_tokens=True).strip()
                
                # Post-process corrections
                correction = self._fix_contractions(correction)
                correction = self._fix_punctuation(correction)
                correction = self._fix_capitalization(correction)
                correction = self._fix_grammar(correction)
                
                # Final cleanup
                correction = correction.strip()
                correction = self.re.sub(r'\s+([.,!?])', r'\1', correction)
                correction = self.re.sub(r'([.,!?])\1+', r'\1', correction)
                
                corrected.add(correction)

            return corrected
        else:
            print("Model is not loaded")  
            return None

    def highlight(self, orig, cor):
        edits = self._get_edits(orig, cor)
        orig_tokens = orig.split()
        ignore_indexes = []

        for edit in edits:
            edit_type = edit[0]
            edit_str_start = edit[1]
            edit_spos = edit[2]
            edit_epos = edit[3]
            edit_str_end = edit[4]

            for i in range(edit_spos+1, edit_epos):
                ignore_indexes.append(i)

            if edit_str_start == "":
                if edit_spos - 1 >= 0:
                    new_edit_str = orig_tokens[edit_spos - 1]
                    edit_spos -= 1
                else:
                    new_edit_str = orig_tokens[edit_spos + 1]
                    edit_spos += 1
                if edit_type == "PUNCT":
                    st = "<a type='" + edit_type + "' edit='" + edit_str_end + "'>" + new_edit_str + "</a>"
                else:
                    st = "<a type='" + edit_type + "' edit='" + new_edit_str + " " + edit_str_end + "'>" + new_edit_str + "</a>"
                orig_tokens[edit_spos] = st
            elif edit_str_end == "":
                st = "<d type='" + edit_type + "' edit=''>" + edit_str_start + "</d>"
                orig_tokens[edit_spos] = st
            else:
                st = "<c type='" + edit_type + "' edit='" + edit_str_end + "'>" + edit_str_start + "</c>"
                orig_tokens[edit_spos] = st

        for i in sorted(ignore_indexes, reverse=True):
            del(orig_tokens[i])

        return " ".join(orig_tokens)

    def _get_edits(self, orig, cor):
        orig = self.annotator.parse(orig)
        cor = self.annotator.parse(cor)
        alignment = self.annotator.align(orig, cor)
        edits = self.annotator.merge(alignment)

        if len(edits) == 0:  
            return []

        edit_annotations = []
        for e in edits:
            e = self.annotator.classify(e)
            edit_annotations.append((e.type[2:], e.o_str, e.o_start, e.o_end, e.c_str, e.c_start, e.c_end))
                
        if len(edit_annotations) > 0:
            return edit_annotations
        else:    
            return []

    def get_edits(self, orig, cor):
        return self._get_edits(orig, cor)