import codecs
import hashlib
import os
import time
import sentencepiece as spm
from mecab_func import check_valid_token_size


class SentencePieceMaker:
    """
    - sentencepieceはbyte pair encoding を言語処理に応用したアルゴリズム
    - 対象の長文に含まれる連続した文字列を単語的なものとして指定された語数になるように結合する
    - 単語の長さは最小１最大は制限なし
        e.g. 英語で最大語数30を指定すると、概ねアルファベット1文字を単語とする
            30以上を指定すると、文字を組み合わせて結合した言葉も単語とする
    - 頻出する文字の連なりを単語と見なすので、文法的には文節や単語の途中までを単語とするのも多い
        e.g. "私は", "低圧電源の" など接頭詞や助詞をよく巻き込む
        e.g. "織田信", "長", "織田信", "広"など頻出する部分を独立させたがる
    """

    def __init__(self):
        self.path_to_save = "placeholder"
        self.sentence_combined = "placeholder"
        self.sp = spm.SentencePieceProcessor()
        self.num_valid_tokens = -999
        self.model_name = "placeholder"

    def set_save_directory(self, path_to_save="") -> bool:
        """
        save directoryの指定と作成
        :param path_to_save: str, save先
        :return: True or None
        """
        if path_to_save == "":
            path_to_save = f"sentence_piece_{hashlib.sha1(time.ctime().encode()).hexdigest()}/"

        if not path_to_save.endswith("/"):
            path_to_save += "/"

        self.path_to_save = path_to_save.replace("\\", "/")
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)
        return True

    def set_sentences(self, sentences: list) -> bool:
        """
        対象文書群れのセット
        文字列をiterableな塊で渡す
        e.g. sentences = ["this is fine today", "today is fine day", "and so on"]

        WARNING
        1文の長さは最大4096とする
        理論上の制約はないとsentencepieceの開発者も言っているし、制約解除オプションもあるが今回は4096
        4096以上のは無視され、あまりいも長いとエラーで止まる
        1文の定義は文頭から改行ないし文末まで
        文中に\nが混ざるとそこまで

        :param sentences:list of string,
        :return:
        """
        self.sentence_combined = "\n".join(sentences)
        return True

    def _set_sentence_directory(self, sentence: str) -> bool:
        self.sentence_combined = sentence
        return True

    def calculate_token_size_with_mecab(self):
        """
        set_sentencesのあとに実施すべし

        文書全体の語数を指定する必要があるが、見当もつかないときは形態素解析機MeCabを使って概算見積もり
        語数が分かる場合は実施しない

        MeCabで使用する辞書はmecab-ipadic-neologdを想定する
        デフォルトの辞書だと、無意味に細かい単語が増えて語数が増加する
        neologdを使えない場合はspacy, ginza, sudachipyで検索してginzaで代用しても良い
        :return:int 概算語数
        """
        return check_valid_token_size(self.sentence_combined)

    def save_text_for_sentence_piece_train(self) -> bool:
        """
        sentencepieceのapiで入力に1行1文のテキストファイルが要求されるので作成
        :return:bool
        """
        with codecs.open(f"{self.path_to_save}delete_me_sentence_file.txt", "w", "utf8") as f:
            f.write(self.sentence_combined)
        return True

    def make_sentence_piece(self, file_name_input_sentence_text: str, num_token_size: int) -> bool:
        """
        sentencepieceのモデルを作成する
        neologd & mecab でも良いのだけれど、neologdは科学・工業ドメインで使うと副作用がひどい
        また辞書に入っていない未知語問題がつきまとう
        今回は専門用語辞書を作りたくないので、sentencepieceによるtoken化を採用
        なお、sentencepieceにも、文節や単語の部分文字列を単語と見なす副作用がある
        そもそもsentencepieceは機械翻訳のためのLSTMの前処理として使うことが想定されており、
        語順を反映しない後処理と相性が良くないという課題がある
        この後段に部分文字列も処理してくれるfasttextを繋げる予定であり、ある程度ごまかされると期待する
        :param file_name_input_sentence_text:テキストファイルのパス, 4096文字板の文を\nで結合
        :param num_token_size:int, sentencepieceが抽出する語数
        :return:
        """
        self.model_name = f"{self.path_to_save}sentence_piece"
        query = f"--input={file_name_input_sentence_text} --model_prefix={self.model_name} --vocab_size={num_token_size}"
        # result auto save
        sp = spm.SentencePieceTrainer.Train(query)

        return True

    def wrapper_with_sentence_list(self, sentences: list, num_token_size: int, magic_number=2.0,
                                   path_to_save="") -> bool:
        """
        list of str からsentencepieceモデルの生成をする

        語数の想像がつかないときはnum_token_sizeを0にするとmecab経由でそれなりの語数を概算

        num_token_sizeは実行後のsentence_piece.vocabの単語風のものを見ながら試行錯誤
        num_token_size, magic_numberを調整
        - 細切れ感がある: 語数を増やす
        - 妙に長い: 語数を減らす
        :param sentences:list of str
        :param num_token_size:int sentencepieceの上限語数, 0指定でMeCab経由でそれらしい数を計算
        :param magic_number:float 個人の経験上文章が短いときはmagic_numberを4位にすると怪しいtokenができにくい
        :param path_to_save: str, セーブ先のフォルダパス
        :return:
        """
        # set save dir
        self.set_save_directory(path_to_save=path_to_save)
        # set sentence
        self.set_sentences(sentences=sentences)
        # calculate token size
        if num_token_size == 0:
            num_token_size = int(self.calculate_token_size_with_mecab() * magic_number)
        # sentencepiece input file, sentences -> string -> text file
        # ## sentencepiece needs text file of imput sentences
        file_name_sentence_piece_input = f"{self.path_to_save}delete_me_sentence_file.txt"
        with codecs.open(file_name_sentence_piece_input, "w", "utf8") as f:
            f.write(self.sentence_combined)
        # make sentence piece model, auto save by api
        self.make_sentence_piece(file_name_input_sentence_text=file_name_sentence_piece_input,
                                 num_token_size=num_token_size)
        # prepare sentencepiece model
        # ## self.model_name はself.make_sentence_pieceでsentencepiece.modelにリネームされる
        # ## sentencepieceのtrain済みモデルの内.modelの方
        self.prepare_tokenizer(f"{self.model_name}.model")

        return True

    def prepare_tokenizer(self, file_name_sentence_piece_model: str) -> bool:
        """
        sentence piece modelのload, 基本的には蛇足なので不要
        :param self:
        :param file_name_sentence_piece_model:sentencepieceのモデルファイル(.model)
        :return:
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(file_name_sentence_piece_model)

        return True

    def tokenize(self, sentence) -> list:
        """
        sentence -> list of token
        :param sentence: str
        :return: list of str
        """
        return self.sp.EncodeAsPieces(sentence)
