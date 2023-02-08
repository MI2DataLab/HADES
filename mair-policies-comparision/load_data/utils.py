import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
from spacy.tokens import Doc, Token
from PyPDF2 import PdfFileReader

def process_tokens(
    doc: pd.Series, nlp: spacy.language.Language, stop_words: List[str]
) -> List[Token]:
    spacy_text = nlp(doc)
    return [
        token
        for token in spacy_text
        if not any([token.is_stop, token.is_punct, token.lemma_.lower() in stop_words, not token.is_alpha])
    ]

def get_filtered_tokens(spacy_text: Doc, stop_words: List[str]) -> List[Token]:
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
        try:
            pageObj = fileReader.getPage(toc_page)
        except:
            return "", -1
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
            paragaph_without_spaces = paragraph.replace(" ", "")
            paragraph_line = [line.replace(" ", "") for line in lines if paragaph_without_spaces in line.replace(" ", "")]
            if len(paragraph_line) == 0:
                continue
            paragraph_line = paragraph_line[0]
            paragraph_line_without_spaces = paragraph_line.replace(" ", "")
            paragaph_without_spaces = paragraph.replace(" ", "")
            try:
                page_str = re.sub(
                            "[^0-9]+",
                            "",
                            paragraph_line_without_spaces[paragraph_line_without_spaces.find(paragraph) + len(paragraph) :],
                        )
                if page_str == '':
                    page_str = 999
                start_page = (
                    int(
                        page_str
                    )
                    + pages_shift
                )
            except Exception as e:
                continue
            if len(rows["start_page"]) > 0:
                rows["end_page"].append(start_page)
                rows["end_text"].append(paragraph if page_str!=999 else None)
            if key != end_paragraph:
                rows["paragraph"].append(key)
                rows["start_page"].append(start_page)
                rows["start_text"].append(paragraph)
            else:
                break
    if len({len(i) for i in rows.values()}) != 1:
        rows["end_page"].append(999)
        rows["end_text"].append(None)
    return pd.DataFrame(rows)


def read_pages_from_pdf(path: str, start_page: int, end_page: int) -> str:
    file = open(path, "rb")
    fileReader = PdfFileReader(file)
    text = ""
    count = start_page - 1
    while count < end_page:
        try:
            pageObj = fileReader.getPage(count)
            count += 1
            text += pageObj.extractText().replace("\n", "")
        except IndexError:
            break
    return text


def read_paragraphs(df: pd.DataFrame, path: str, country: str, root: str = "") -> pd.DataFrame:
    result_dict = {"paragraph": [], "country": [], "text_path": []}
    for i, row in df.iterrows():
        file_name = row.paragraph.replace(":","").replace(" ","_").replace(",","").replace("/","_").replace("(","").replace(")","").replace("&","").replace("-","_").replace("__","_").lower()
        txt_destination = f"{root}{country}_{file_name}.txt"
        if row.start_page is None:
            text = ""
        else:
            start_page = row.start_page
            end_page = row.end_page
            for i in [0, -1, 1, 2]:
                start_page = row.start_page + i
                text = read_pages_from_pdf(path, start_page, end_page)
                if text != "":
                    break
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
