# analysis_template

## Docker立ち上げ方法

1. `docker compose up -d --build` (イメージ作成→コンテナ作成→コンテナ起動)
2. `docker image ls` (作成されたイメージの確認)
3. `docker container ls` (現在走っているコンテナの確認)
4. `docker compose exec python3 bash` (コンテナへ接続)
5. `python sample.py 180.0` (sample.pyの実行)
6. `exit` (コンテナからの接続を切る)

- `docker compose down` (コンテナを終了し、削除)
- `docker stop {containerid}` (コンテナ停止)
- `docker rm {containerid}` (コンテナ削除)
- `docker-compose down --rmi all --volumes --remove-orphans` (コンテナ、イメージ、ボリューム、ネットワーク、未定義コンテナ、全てを一括消去)
- `docker compose up -d` (コンテナ再起動)
- `docker image rm {imageid}` (イメージ削除)
- `docker run -v $PWD/src:/root/src -w /root/src -it --rm -p 7777:8888 docker-python-python3 jupyter-lab --ip 0.0.0.0 --allow-root -b localhost` (Jupyter Notebook立ち上げ)　<http://127.0.0.1:7777>

## ディレクトリ構成

- src (コンテナがマウントするディレクトリ)
  - input (train.csv, test.csvなどの入力ファイルを入れるフォルダ)
  - code (計算用のコードのフォルダ)
  - code-analysis (分析用のコードやJupyter Notebookのフォルダ)
  - model (モデルや特徴量を保存するフォルダ)
  - submission (提出用ファイルを保存するフォルダ)
- Dockerfile
- docker-compose.yml
- requirements.txt
- .gitignore
- README.md
- .devcontainer
  - devcontainer.json (VScodeでリモートアクセスするための設定ファイル)

## VScodeでDockerコンテナに接続

1. 拡張機能の [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)をインストール
2. 左下の緑のボタンを押して**Reopen in Container**を選択
3. 各種設定は `.devcontainer/devcontainer.json`に書き込む
4. リモート接続したコンテナには拡張機能が入っていないので必要に応じて[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)等をインストールする
