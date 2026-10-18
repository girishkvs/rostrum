"""Word normalization used to compare spoken sentences."""
import re
import unicodedata

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# (suffix, the word must be longer than this, replacement), first match wins
SUFFIX_RULES = (("ied", 4, "y"), ("ing", 5, ""), ("ed", 4, ""), ("ly", 5, ""))
THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3})")
CLOCK_TIME = re.compile(r"(\d{1,2})[:.](\d\d)\b")
DIGITS = re.compile(r"\d+")
NOT_ASCII_WORD = re.compile(r"[^a-z0-9' ]+")
# Unicode categories kept inside words: letters, numbers and marks (accents, vowel signs).
WORD_CATEGORIES = ("L", "N", "M")


class NumberWords:
    """Spells out numbers, so "42" in a transcript matches "forty two" in a script."""

    def say(self, number):
        if number < 20:
            return ONES[number]
        if number < 100:
            return TENS[number // 10] + ("" if number % 10 == 0 else " " + ONES[number % 10])
        if number < 1000:
            return ONES[number // 100] + " hundred" + ("" if number % 100 == 0 else " " + self.say(number % 100))
        if number < 1000000:
            return self.say(number // 1000) + " thousand" + ("" if number % 1000 == 0 else " " + self.say(number % 1000))
        return str(number)


class Tokenizer:
    """Lower-case word tokens. Clock times, percents and numbers are spelled out; punctuation is dropped. Letters
    outside ASCII (accents, other scripts) stay part of their word."""

    def __init__(self):
        self.numbers = NumberWords()

    def tokens(self, raw):
        text = unicodedata.normalize("NFC", raw).strip().casefold().replace("\u2019", "'").replace("\u2018", "'")
        text = THOUSANDS_COMMA.sub("", text)
        text = CLOCK_TIME.sub(self.clock, text)
        text = text.replace("%", " percent ")
        text = DIGITS.sub(self.number, text)
        text = self.word_characters(text)
        return [token for token in (part.strip("'") for part in text.split()) if token]

    def word_characters(self, text):
        """Letters, numbers, marks and apostrophes stay; everything else becomes a space."""
        if text.isascii():
            return NOT_ASCII_WORD.sub(" ", text)
        return "".join(character if self.in_word(character) else " " for character in text)

    def in_word(self, character):
        return (character == "'" or
                unicodedata.category(character)[0] in WORD_CATEGORIES)

    def clock(self, match):
        hours = int(match.group(1))
        minutes = match.group(2)
        tail = ("oh " + ONES[int(minutes)]) if minutes.startswith("0") else self.numbers.say(int(minutes))
        return f" {self.numbers.say(hours)} {tail} "

    def number(self, match):
        digits = match.group(0)
        if len(digits) <= 6:
            return " " + self.numbers.say(int(digits)) + " "
        return digits

    def text_tokens(self, text):
        return [token for word in text.split() for token in self.tokens(word)]


class Stemmer:
    def stem(self, token):
        for suffix, longer_than, replacement in SUFFIX_RULES:
            if (len(token) > longer_than and
                    token.endswith(suffix)):
                token = token[:-len(suffix)] + replacement
                break
        if (len(token) > 3 and
                token.endswith("s") and
                not token.endswith("ss")):
            token = token[:-1]
        if (len(token) > 4 and
                token.endswith("e")):
            token = token[:-1]
        return token


class ContentWords:
    """The key words of a phrase: stemmed, without stop words, fillers or one-letter tokens."""

    def __init__(self, stop_words, filler_words):
        self.stop_words = frozenset(stop_words)
        self.filler_words = frozenset(filler_words)
        self.stemmer = Stemmer()

    def of(self, tokens):
        return {self.stemmer.stem(token) for token in tokens if self.is_content(token)}

    def is_content(self, token):
        return (token not in self.stop_words and
                token not in self.filler_words and
                len(token) > 1)
