import MeCab
import joblib
from more_itertools import chunked
import os

"""
- MeCab使用上の注意 for win user
    - 日本語は単語区切りが陽に現れない膠着言語なので、文を単語にする際は形態素解析が必要
      e.g. this is fine today, 本日は晴天なり
            ヨーロッパ言語はスペースで単語を区切るが、日本語は連続した文字列なのでどこが切れ目かは解析が必要になる
    - windowsでpython & mecab を使おうとするとインストールが猛烈に面倒くさいので以下を推奨
        - cygwin, make等と聞いて「あぁアレね」と思える人々
            適宜ググると参考pageが見つかるので、それに従うこと
            弊社内ネットワークは高確率で業界デファクトレベルのリソースすらアクセス遮断するので、自宅から送ること
        - 「コンピューターはわからない」という人々
            pypiにサクッとインストールできるコンパイル済みモジュールが上がっているのでpipでインストール
            pip install mecab-python-windows
            池上 有希乃氏が作成と管理をしてくれているモジュールで、ご本人は信じられる方ですが、
            他人がコンパイルした謎ファイルをウイルスチェックなしにインストールするのが何とも
    - linux userは何も考えずにapt, brewなどでよしなに。日本言語処理界隈のほぼすべての人がコレを使うので、信頼して良いと思われる
- 辞書mecab-ipadic-neologd
    -　形態素解析は、辞書と呼ばれる単語＆よくある組み合わせ一覧を参照する
        辞書が貧弱だと形態素解析が失敗しやすい
        neologdはLINE株式会社の佐藤氏提供の一般用語陽辞書 (https://github.com/neologd/mecab-ipadic-neologd
        LINEに頻出する言葉を中心に収録しているので、専門用語の多い科学業界ではコレを使うと副作用がひどいことがある
    - linux userは何も考えずにapt, brewなどでよしなに。日本言語処理界隈のほぼすべての人がコレを使うので、信頼して良いと思われる
    - windows user は何とかしてlinuxにアクセスする必要がある
        - linux にアクセスできる方
            - linuxでneologdをインストールして、windowsにコピー
            - 以下のファイルのdicdir変数で辞書の参照先ホルダを指定しているので、コピーしたフォルダを指定
                %MeCabをインストールしたフォルダ%/etc/mecabrc
        - linuxにアクセスできない方
            - linux必須のため、virtualboxやwindows subsystem for linuxなどを使ってlinuxにアクセスする
            - 弊社の方であればご連絡頂いたら社内LANで送れます

"""

# MeCabインスタンスはオーバーヘッドが大きいし、使い回すとたまにおかしくなるためココで生成
tagger = MeCab.Tagger("")
# for speed up KUROMAJUTU, no meaning
tagger.parse("ダミー処理")


def token_ripper(mecab_parsed: str) -> list:
    """
    MeCab.Tagger().parse(sentence)の出力からtokenの原形を収集
    MeCabの辞書はNeologdを使う前提で、以下のフォーマット
        活用あり表層形\t品詞や読みなどの情報
    品詞や読みなどの情報は','区切りで6番目に原形が入る
    原形未登録の場合は*が入っている
    :param mecab_parsed:
    :return:
    """
    lines = mecab_parsed.split('\n')
    _ = lines.pop(-1)  # blank
    _ = lines.pop(-1)  # EOS
    tokens = []
    for line in lines:
        surface, sp = line.split("\t")
        sp = sp.split(',')
        tokens.append(surface if sp[6] == "*" else sp[6])

    return tokens


def parse_multi_line(lines):
    """"""
    tokens = []
    _ = [tokens.extend(set(token_ripper(tagger.parse(line)))) for line in lines]
    return set(tokens)


def check_valid_token_size(document_for_sentence_piece: str) -> int:
    """
    入力された文字列@日本語の異なり語数を返す
    長文を形態素解析すると、単語選択の選択肢を保持し続けるのでメモリ的に厳しい
    適当に分割して並列処理に流す
    :param document_for_sentence_piece:
    :return:
    """
    document_for_sentence_piece = document_for_sentence_piece.replace("\r\n", "\n")
    if document_for_sentence_piece.count('\n') > 0:
        lines = document_for_sentence_piece.split('\n')
    elif document_for_sentence_piece.count('。') > 0:
        lines = document_for_sentence_piece.split('。')
    else:
        lines = document_for_sentence_piece.split(".")

    workers = os.cpu_count()
    chunks = chunked(lines, workers)
    result = joblib.Parallel(n_jobs=workers)([
        joblib.delayed(parse_multi_line)(chunk)
        for chunk in chunks
    ])
    tokens = []
    _ = [tokens.extend(res) for res in result]
    return len(set(tokens))
