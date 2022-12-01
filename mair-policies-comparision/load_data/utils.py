import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
from PyPDF2 import PdfFileReader


def process_tokens(
        doc: pd.Series, nlp: spacy.language.Language, stop_words: List[str]
) -> List[str]:
    spacy_text = nlp(doc)
    return [
        token
        for token in spacy_text
        if not any([token.is_stop, token.is_punct, token.lemma_.lower() in stop_words, not token.is_alpha])
    ]


def process_lemmas(doc: pd.Series) -> List[str]:
    return [token.lemma_.lower() for token in doc]


def _multiply_ngrams(tokens: List[str]):
    for token in tokens:
        if " " in token:
            yield token
            yield token
        yield token


def get_table_of_contents(path: str, toc: str = "Table of Contents") -> Tuple[str, int]:
    file = open(path, "rb")
    fileReader = PdfFileReader(file)
    text = ""
    toc_page = 0
    while not toc in text:
        pageObj = fileReader.getPage(toc_page)
        text = pageObj.extractText()
        toc_page += 1
    file.close()
    return text, toc_page


def get_paragraphs_df(
        toc: str, pages_shift: int, paragraphs_names: Dict[str, List[str]], end_paragraph: str
) -> pd.DataFrame:
    lines = toc.split("\n")
    rows = {"paragraph": [], "start_page": [], "end_page": [], "start_text": [], "end_text": []}
    for key, paragraphs in paragraphs_names.items():
        for paragraph in paragraphs:
            paragraph_line = [line for line in lines if paragraph in line]
            if len(paragraph_line) == 0:
                continue
            paragraph_line = paragraph_line[0]
            try:
                start_page = (
                        int(
                            re.sub(
                                "[^0-9]+",
                                "",
                                paragraph_line[paragraph_line.find(paragraph) + len(paragraph):],
                            )
                        )
                        + pages_shift
                )
            except:
                break
            if len(rows["start_page"]) > 0:
                rows["end_page"].append(start_page)
                rows["end_text"].append(paragraph)
            if key != end_paragraph:
                rows["paragraph"].append(key)
                rows["start_page"].append(start_page)
                rows["start_text"].append(paragraph)
    return pd.DataFrame(rows)


def read_pages_from_pdf(path: str, start_page: int, end_page: int) -> str:
    file = open(path, "rb")
    fileReader = PdfFileReader(file)
    text = ""
    count = start_page - 1
    while count < end_page:
        pageObj = fileReader.getPage(count)
        count += 1
        text += pageObj.extractText().replace("\n", "")
    return text


def read_paragraphs(df: pd.DataFrame, path: str, country: str, root: str = "") -> pd.DataFrame:
    result_dict = {"paragraph": [], "country": [], "text_path": []}
    for i, row in df.iterrows():
        txt_destination = f"{root}{country}_{row.paragraph}.txt"
        if row.start_page is None:
            text = ""
        else:
            start_page = row.start_page
            end_page = row.end_page
            text = read_pages_from_pdf(path, start_page, end_page)
            if row.start_text is not None:
                try:
                    text = row.start_text + text.split(row.start_text, 1)[1]
                except:
                    pass
            if row.end_text is not None:
                text = text.split(row.end_text, 1)[0]
        text_file = open(txt_destination, "w+", encoding="utf-8")
        n = text_file.write(text)
        text_file.close()
        result_dict["paragraph"].append(row.paragraph)
        result_dict["country"].append(country)
        result_dict["text_path"].append(txt_destination)
    return pd.DataFrame(result_dict)


def process_all_documents(
        directory_path: str,
        paragraphs_names: Dict[str, List[str]],
        save_txt: str,
        end_paragraph: str,
        toc_str: str = "Table of Contents",
        pages_shift: Optional[int] = None,
) -> pd.DataFrame:
    """Process documents from directory_path with
    table of contents with paragraph names and pages

    Args:
        directory_path (str): directory with documents to process
        paragraphs_names (Dict[str, List[str]]): key - name of pargraph that should
        be displayed in the final df, value - list of possible names of this paragraph in toc
        save_txt (str): path to directory where txt files should be saved
        end_paragraph (str): last paragraph of the text that should not be present in final df
        toc_str (str, optional): name of table of contents in documents. Defaults to "Table of Contents".
        pages_shift (int, optional): difference between page number in table of contents and in pdf file.
            Defaults to None which will be interpreted as pages_shift = page of toc

    Returns:
        pd.DataFrame: data frame with desired format
    """
    dir_list = os.listdir(directory_path)
    dir_list = [file for file in dir_list if file[-3:] == "pdf"]
    df = pd.DataFrame({"paragraph": [], "country": [], "text_path": []})
    for doc in dir_list:
        toc, toc_page = get_table_of_contents(directory_path + doc, toc_str)
        paragraphs_df = get_paragraphs_df(
            toc, pages_shift or toc_page, paragraphs_names, end_paragraph
        )
        doc_df = read_paragraphs(paragraphs_df, directory_path + "/" + doc, doc[:-4], save_txt)
        df = pd.concat([df, doc_df], ignore_index=True)
    return df


def text_cleaning(text):
    # deleting URLs
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE)
    # deleting headlines
    text = re.sub(r'((\d\.)+\d) +[A-Z]([a-z]|\s|,)+', '', text)
    # deleting random numbers
    text = re.sub(r'((\d)+ +)+\(\d+\)', '', text)
    # deleting picture descriptions
    text = re.sub(r' \d+ [A-Z](\w|\s|,)+.', '', text)
    # deleting tables
    sentences = text.split('. ')
    to_delete = False
    sentences_copy = sentences.copy()
    for i, sentence in enumerate(sentences):
        if to_delete:
            to_delete = False
            sentences_copy[i] = ''
        if re.match(r'\s+Table \d+', sentence):
            to_delete = True
            sentences_copy[i] = ''
    text = '. '.join(sentences_copy)
    # deleting multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r' . ', '', text)
    return text
