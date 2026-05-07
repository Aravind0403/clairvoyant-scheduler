// features.go — mirrors feature_extractor.py (runtime version, not featurize.py).
//
// Feature order must match train.py exactly:
//   6 numeric: prompt_token_len, has_code_keyword, has_length_constraint,
//              ends_with_question, has_format_keyword, clause_count
//  13 one-hot: verb_what, verb_write, verb_explain, verb_summarize, verb_how,
//              verb_list, verb_implement, verb_compare, verb_describe,
//              verb_generate, verb_why, verb_define, verb_other
//
// Token counting uses len(text)/4 (same approximation as feature_extractor.py).
// Training data used bert-base-uncased counts — this introduces a small
// distribution shift for prompt_token_len but is acceptable for Phase 3.

package predictor

import (
	"regexp"
	"strings"
	"unicode"
)

// NFeatures is the total input width: 6 numeric + 13 verb one-hot.
const NFeatures = 19

var (
	codeKeywords = map[string]bool{
		// original set
		"code": true, "function": true, "def": true, "class": true,
		"script": true, "program": true, "implement": true, "algorithm": true,
		"debug": true, "refactor": true, "import": true, "variable": true,
		"loop": true, "recursion": true, "api": true, "sql": true, "regex": true,
		// web / framework keywords common in ShareGPT code requests
		"react": true, "angular": true, "vue": true, "nextjs": true,
		"typescript": true, "javascript": true, "html": true, "css": true,
		"node": true, "express": true, "django": true, "flask": true,
		"component": true, "endpoint": true, "middleware": true, "query": true,
	}

	constraintRe = regexp.MustCompile(
		`(?i)\bin\s+\d+\s+words?\b` +
			`|\bin\s+\d+\s+sentences?\b` +
			`|\bno\s+more\s+than\s+\d+` +
			`|\bunder\s+\d+\s+words?\b` +
			`|\bbriefly\b` +
			`|\bconcisely\b` +
			`|\bshortly\b` +
			`|\bone[\s-]liner\b` +
			`|\btl;?dr\b`,
	)

	formatKeywords = []string{
		"list", "table", "bullet", "step by step", "enumerate",
		"outline", "numbered", "format", "structure",
	}

	// verbPatterns: first match wins (same priority order as feature_extractor.py).
	verbPatterns = []struct {
		label string
		re    *regexp.Regexp
	}{
		{"summarize", regexp.MustCompile(`(?i)\bsummariz(?:e|ing)\b`)},
		{"explain", regexp.MustCompile(`(?i)\bexplain\b`)},
		{"compare", regexp.MustCompile(`(?i)\bcompar(?:e|ing)\b`)},
		{"translate", regexp.MustCompile(`(?i)\btranslat(?:e|ing)\b`)},
		{"generate", regexp.MustCompile(`(?i)\bgenerat(?:e|ing)\b`)},
		{"implement", regexp.MustCompile(`(?i)\bimplement\b`)},
		{"debug", regexp.MustCompile(`(?i)\bdebug\b`)},
		{"refactor", regexp.MustCompile(`(?i)\brefactor\b`)},
		{"list", regexp.MustCompile(`(?i)\blist\b`)},
		{"write", regexp.MustCompile(`(?i)\bwrite\b`)},
		{"describe", regexp.MustCompile(`(?i)\bdescrib(?:e|ing)\b`)},
		{"define", regexp.MustCompile(`(?i)\bdefin(?:e|ing)\b`)},
		{"what", regexp.MustCompile(`(?i)\bwhat\b`)},
		{"how", regexp.MustCompile(`(?i)\bhow\b`)},
		{"why", regexp.MustCompile(`(?i)\bwhy\b`)},
	}

	// knownVerbs must match KNOWN_VERBS order in train.py exactly.
	// This order maps directly to the 13 one-hot columns.
	knownVerbs = []string{
		"what", "write", "explain", "summarize", "how",
		"list", "implement", "compare", "describe",
		"generate", "why", "define", "other",
	}
)

// Features holds the 19 float32 values fed into the ONNX model.
type Features struct {
	PromptTokenLen      float32
	HasCodeKeyword      float32
	HasLengthConstraint float32
	EndsWithQuestion    float32
	HasFormatKeyword    float32
	ClauseCount         float32
	VerbOneHot          [13]float32
}

// ToSlice returns features as a flat []float32 in model-input order.
func (f Features) ToSlice() []float32 {
	s := make([]float32, 0, NFeatures)
	s = append(s,
		f.PromptTokenLen,
		f.HasCodeKeyword,
		f.HasLengthConstraint,
		f.EndsWithQuestion,
		f.HasFormatKeyword,
		f.ClauseCount,
	)
	return append(s, f.VerbOneHot[:]...)
}

// Extract derives all features from a raw prompt string.
func Extract(prompt string) Features {
	lower := strings.ToLower(prompt)
	words := extractWords(lower)

	// ── numeric features ────────────────────────────────────────────────────

	tokenLen := float32(len(prompt) / 4)

	hasCode := float32(0)
	for _, w := range words {
		if codeKeywords[w] {
			hasCode = 1
			break
		}
	}

	hasConstraint := float32(0)
	if constraintRe.MatchString(prompt) {
		hasConstraint = 1
	}

	trimmed := strings.TrimRightFunc(prompt, unicode.IsSpace)
	endsQ := float32(0)
	if len(trimmed) > 0 && trimmed[len(trimmed)-1] == '?' {
		endsQ = 1
	}

	hasFormat := float32(0)
	for _, kw := range formatKeywords {
		if strings.Contains(lower, kw) {
			hasFormat = 1
			break
		}
	}

	clauseCnt := float32(
		strings.Count(prompt, ",") +
			strings.Count(prompt, ";") +
			strings.Count(lower, " and ") +
			strings.Count(lower, " but ") +
			strings.Count(lower, " because "),
	)

	// ── verb one-hot ─────────────────────────────────────────────────────────

	verb := "other"
	for _, vp := range verbPatterns {
		if vp.re.MatchString(prompt) {
			verb = vp.label
			break
		}
	}

	var verbOneHot [13]float32
	for i, v := range knownVerbs {
		if v == verb {
			verbOneHot[i] = 1
			break
		}
	}

	return Features{
		PromptTokenLen:      tokenLen,
		HasCodeKeyword:      hasCode,
		HasLengthConstraint: hasConstraint,
		EndsWithQuestion:    endsQ,
		HasFormatKeyword:    hasFormat,
		ClauseCount:         clauseCnt,
		VerbOneHot:          verbOneHot,
	}
}

// extractWords splits lowercase text into word tokens (letters/digits only).
func extractWords(text string) []string {
	var words []string
	var cur strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(r)
		} else if cur.Len() > 0 {
			words = append(words, cur.String())
			cur.Reset()
		}
	}
	if cur.Len() > 0 {
		words = append(words, cur.String())
	}
	return words
}
