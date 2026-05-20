import json
import re


class Extract:
    def __init__(self, string):
        self.raw_str = string

    def extract_code(self):
        """Extract generated heuristic code and inertia weight from an LLM response."""
        data = self._parse_json_object(self.raw_str)
        if data is not None:
            code = data.get("code")
            inertia_weight = data.get("inertia_weight")
            if code is not None and inertia_weight is not None:
                try:
                    return str(code), float(inertia_weight)
                except (TypeError, ValueError):
                    pass

        code_match = re.search(r'"code":\s*"((?:\\.|[^"\\])*)"', self.raw_str)
        inertia_match = re.search(r'"inertia_weight":\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', self.raw_str)
        if not code_match or not inertia_match:
            print("Failed to extract inertia weight and executable code")
            return None

        try:
            escaped_code = code_match.group(1)
            code = escaped_code.encode().decode("unicode_escape")
            inertia_weight = float(inertia_match.group(1))
            return code, inertia_weight
        except Exception:
            print("Failed to extract inertia weight and executable code")
            return None

    def save_code(self, path, code):
        """Save generated heuristic code into a Python file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"---The code has saved in {path}---")
        except Exception:
            print("---Runtime Error: failed to save the code in corresponding file---")

    def recoder(self):
        pass

    def extract_params(self):
        try:
            data = json.loads(self.raw_str.strip())
            w = float(data.get("w", 0.9))
            c1 = float(data.get("c1", 2.0))
            c2 = float(data.get("c2", 2.0))
            explanation = data.get("explanation", "LLM suggestion")
        except Exception:
            print("LLM returned invalid JSON; using fallback parameters")
            w, c1, c2, explanation = 1.1, 2.5, 0.3, "fallback exploration strategy"

        print(f"LLM diagnosis: {explanation}")
        print(f"LLM parameters: w={w:.4f}, c1={c1:.3f}, c2={c2:.3f}")
        return w, c1, c2

    @staticmethod
    def _parse_json_object(text):
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
