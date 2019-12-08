"""
Corpus reader for the "Hyperpartisan News Detection" dataset,
from SemEval-2019 Task4.

Reads the Corpus (from structured XML files) into memory.
"""

import xml.sax
import lxml.sax, lxml.etree
import re
import concurrent

class XmlDataExtractor(xml.sax.ContentHandler):
    """
    Generic XML data extractor.
    """

    def __init__(self):
        super().__init__()
        self.data = dict()
        self.lxmlhandler = None

    def get_data(self):
        return self.data


class NewsExtractor(XmlDataExtractor):
    """
    Parsing xml units with xml.sax + lxml.sax.ElementTreeContentHandler.
    Parser used in ../scripts/semeval-pan-2019-tf-extractor.py
    """

    def __init__(self):
        super().__init__()
        self.lxmlhandler = None

    def startElement(self, name, attrs):
        if name == 'articles':
            return
        if name == 'article':
            self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
        self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler is not None:
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler is not None:
            self.lxmlhandler.endElement(name)
            if name =='article':
                # complete article parsed
                article = NewsArticle(self.lxmlhandler.etree.getroot())
                self.data[article.get_id()] = article
                self.lxmlhandler = None


class NewsExtractorFeaturizerFromStream(XmlDataExtractor):
    """
    Extracts news data from the XML stream and immediately featurizes
     each article, not needing to keep the whole dataset in memory.
    Featurizes objects concurrently with a ThreadPool.
    """
    ## NOTE already tried multi-threading the featurizer step,
    ##  but due to python's GIP lock it always runs in synch.

    def __init__(self, featurizer):
        super().__init__()
        self.lxmlhandler = None
        self.featurizer = featurizer
        self.counter = 0

    def startElement(self, name, attrs):
        if name == 'articles':
            return
        if name == 'article':
            self.lxmlhandler = lxml.sax.ElementTreeContentHandler()
        self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler is not None:
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler is not None:
            self.lxmlhandler.endElement(name)
            if name =='article':
                # complete article parsed
                self.counter += 1
                if self.counter % 500 == 0:
                    print('Doc. Progress: {:5}'.format(self.counter))

                article = NewsArticle(self.lxmlhandler.etree.getroot())
                self.data[article.get_id()] = self.featurizer(article)
                self.lxmlhandler = None


class GroundTruthExtractor(XmlDataExtractor):
    """
    SAX parser for gound truth XML.
    """

    def __init__(self):
        super().__init__()

    def startElement(self, name, attrs):
        if name == 'article':
            hyperpartisan = attrs.getValue('hyperpartisan')
            bias = attrs.getValue('bias') if 'bias' in attrs else None
            self.data[attrs.getValue('id')] = (hyperpartisan, bias)


class NewsArticle:
    """
    Class representing a single News article from the Hyperpartisan News Corpora.
    """

    # Pattern for replacing non-breaking spaces
    txt_tospace1 = re.compile(r'&#160;')

    # Pattern for replacing numbers
    number_pattern = re.compile(r'([\d]+[\.,]?[\d]*)')

    # Token for representing numbers
    NUMBER_TOKEN = '<num>'

    def __init__(self, rootNode):
        self.root = rootNode
        self.text = ' '.join([lxml.etree.tostring(el, method='text', encoding='unicode', with_tail=False).strip() \
                              for el in self.root.getiterator(tag='p')])

        self.hyperpartisan = None
        self.bias = None
    
    def get_id(self):
        return self.root.get('id')
    
    def get_title(self):
        return self.clean_text(self.root.get('title'))
    
    def get_text(self):
        return self.clean_text(self.text)

    def get_text_preprocessed(self):
        return self.preprocess_text(self.get_text())

    def get_text_alpha(self):
        return re.sub('[^A-Za-z ]', '', self.text)
    
    def set_ground_truth(self, hyperpartisan, bias):
        self.hyperpartisan = hyperpartisan
        self.bias = bias

    def get_hyperpartisan(self) -> str:
        ## May be None if not yet paired with ground-truth
        return self.hyperpartisan

    def is_hyperpartisan(self) -> bool:
        return self.get_hyperpartisan() == 'true'

    def get_bias(self) -> str:
        ## May be None if dataset was labeled by article (not by publisher)
        return self.bias

    def get_links(self):
        return self.root.iterdescendants(tag='a')


    @classmethod
    def clean_text(cls, text):
        """Clean the text extracted from XML."""
        text = text.replace("&amp;", "&")
        text = text.replace("&gt;", ">")
        text = text.replace("&lt;", "<")
        text = text.replace("<p>", " ")
        text = text.replace("</p>", " ")
        text = text.replace(" _", " ")
        text = text.replace("–", "-")
        text = text.replace("”", "\"")
        text = text.replace("“", "\"")
        text = text.replace("’", "'")
        text = text.replace("\t", " ")
        text = text.replace("⚪", " ")
        text = text.replace("  ", " ")

        text = cls.txt_tospace1.sub(' ', text)
        return text

    @classmethod
    def preprocess_text(cls, text):
        ## Substitute numbers with a special token
        text = cls.number_pattern.sub(cls.NUMBER_TOKEN, text)
        return text
