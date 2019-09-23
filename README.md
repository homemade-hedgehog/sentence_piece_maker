# sentence_piece_maker

### なんで作ったの？
形態素解析辞書の貧弱さに絶望した  
巷では辞書はneologdを使うけど、科学技術などの専門用語が多い界隈では副作用＆未知語がひどくてどうにもならない  
named entityの解析方法は学術の皆様に期待しつつ、とりあえず使えるものを作成

### コレは何？
- sentencepieceのモデルを文のリストから生成するラッパー
- 基本的にはsentencepieceのusageのラッピングをしている
- sentencepieceのtoken数に見当がつかないときのために、mecabで分割したときの異なり語数を算出するオマケがついてる

### usage
```python
from sentence_piece_func import SentencePieceMaker
spm = SentencePieceMaker()

sentences = [
    "'マイクロ化学デバイスおよび解析装置\n本発明は、マイクロ化学デバイスおよび解析装置に係り、特に、細胞を保持するマイクロウエルが多数形成されたマイクロ化学デバイスおよび解析装置に関する。", 
    "刃物ホルダー\n本発明は、例えば複数のウィンナーが連なった連鎖状ウィンナーといった連鎖状食品の結束部を切断する刃物を保持する刃物ホルダーに関するものである。"
]

spm.wrapper_with_sentence_list(sentences=sentences, num_token_size=0, magic_number=2.0)

spm.tokenize(sentences[0]) # '▁', 'マイクロ化学デバイス', 'および解析装置', '▁本発明は', '、', 'マイクロ化学デバイス', 'および解析装置',...
```

### 注意
- sentencepieceの形態素解析の代用扱いは本来の適用範囲を逸脱している
  - そもそも、後段には文を部分文字列に切った系列を入力とするLSTMなどを前提としており、bag of wordsに入れることを想定していない
  - この逸脱を少しでも緩和するために、後段はword2vecじゃなくてfasttext推奨(後日サンプルをアップ予定)
- 基本的な説明はdocstringに書いたのでそちらを参照
  - jupyterユーザー: shift + tab 3連打
  - pycharmユーザー: ctrl + q
- sentencepiece モデルの生成先
  - model: sentence_piece_*/sentence_piece.model
  - vocab: sentence_piece_*/sentence_piece.vocab
- wrapper_with_sentence_listの引数の意味
  - sentences: list of str, 入力文の集合, バリエーションが多いほど妥当な分割になる。対象分野の十分な量の文書など。なければwikidumpなど
  - num_token_size: int, sentencepiece異なり語数, 見当がつかなければ0指定でmecab異なり語数を計算に行く
  - magic_number: float, num_token_size=0のときだけ反映。mecab異なり語数*magic_numberをsentencepieceの異なり語数とする
    - 経験上、十分な文書量なら1で良い。文書数が少ないときは4程度まで
    - num_token, magic_numberを調整すると、sentence_piece.vocabの文字列の切れ方が変わる
      - 細切れ感がある: 大きく
      - 文節に見えるほど長い: 小さく
