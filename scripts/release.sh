#!/bin/bash

# エラーが発生したら即座に終了
set -e

# 色の定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 変更があるか確認
if git diff --quiet && git diff --cached --quiet; then
    echo -e "${RED}変更がありません${NC}"
    exit 1
fi

# コミットタイプの選択
echo -e "${YELLOW}コミットタイプを選択してください:${NC}"
echo "1) feat: 新機能"
echo "2) fix: バグ修正"
echo "3) breaking: 破壊的変更"
read -p "選択 (1-3): " choice

case $choice in
    1)
        TYPE="feat"
        ;;
    2)
        TYPE="fix"
        ;;
    3)
        TYPE="feat"
        IS_BREAKING=true
        ;;
    *)
        echo -e "${RED}無効な選択です${NC}"
        exit 1
        ;;
esac

# コミットメッセージの入力
echo -e "${YELLOW}変更内容を簡潔に説明してください:${NC}"
read -p "> " message

# コミットメッセージの構築
if [ "$IS_BREAKING" = true ]; then
    echo -e "${YELLOW}破壊的変更の詳細を説明してください:${NC}"
    read -p "> " breaking_message
    COMMIT_MESSAGE="$TYPE: $message

BREAKING CHANGE: $breaking_message"
else
    COMMIT_MESSAGE="$TYPE: $message"
fi

# プッシュとPRの作成の確認
echo -e "${YELLOW}プッシュとPRの作成を行いますか？${NC}"
echo "1) はい"
echo "2) いいえ（コミットのみ）"
read -p "選択 (1-2): " push_choice

if [ "$push_choice" = "1" ]; then
    # ブランチ名の生成（コミットメッセージから）
    BRANCH_NAME=$(echo "$message" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-*$//')
    BRANCH_NAME="${TYPE}/${BRANCH_NAME}"

    # 新しいブランチを作成
    echo -e "${GREEN}新しいブランチを作成しています: ${YELLOW}${BRANCH_NAME}${NC}"
    git checkout -b "$BRANCH_NAME"

    # 変更をステージング
    echo -e "${GREEN}変更をステージングしています...${NC}"
    # ルートディレクトリに移動
    cd "$(git rev-parse --show-toplevel)"
    # 全ての変更をステージング
    git add -A

    # 変更があるか確認
    if git diff --cached --quiet; then
        echo -e "${RED}ステージングする変更がありません${NC}"
        exit 1
    fi

    # コミット
    echo -e "${GREEN}コミットを作成しています...${NC}"
    git commit -m "$COMMIT_MESSAGE"

    # プッシュ
    echo -e "${GREEN}変更をプッシュしています...${NC}"
    git push origin "$BRANCH_NAME"

    # プルリクエストの作成
    echo -e "${GREEN}プルリクエストを作成しています...${NC}"

    # PRの説明文を生成
    PR_BODY="## 変更内容

$message

$(if [ "$IS_BREAKING" = true ]; then echo -e "\n## 破壊的変更\n\n$breaking_message"; fi)

## 注意
このプルリクエストがマージされると、GitHub Actionsが自動的にバージョン管理とリリースを行います。

### バージョン管理について
- このPRに含まれる全てのコミットメッセージが解析されます
- 最も大きな変更に基づいてバージョンが更新されます：
  - \`feat:\` → マイナーバージョンアップ（0.1.0 → 0.2.0）
  - \`fix:\` → パッチバージョンアップ（0.1.0 → 0.1.1）
  - \`BREAKING CHANGE:\` → メジャーバージョンアップ（0.1.0 → 1.0.0）
- 複数のコミットがある場合、最も大きな変更が適用されます
- コミットメッセージは[Angularのコミットメッセージ規約](https://www.conventionalcommits.org/ja/v1.0.0/)に従ってください"

    # GitHub CLIが利用可能かチェック
    if command -v gh &> /dev/null; then
        gh pr create \
          --title "$COMMIT_MESSAGE" \
          --body "$PR_BODY" \
          --base main
    else
        echo -e "${YELLOW}GitHub CLI (gh) がインストールされていません。${NC}"
        echo -e "${YELLOW}以下のコマンドでインストールしてください：${NC}"
        echo -e "${BLUE}brew install gh${NC}"
        echo -e "${YELLOW}インストール後、以下のコマンドで認証してください：${NC}"
        echo -e "${BLUE}gh auth login${NC}"
        echo -e "\n${YELLOW}または、以下のURLから手動でプルリクエストを作成してください：${NC}"
        echo -e "${BLUE}https://github.com/YutoNose/dariko/compare/main...${BRANCH_NAME}${NC}"
        echo -e "\n${YELLOW}プルリクエストの説明：${NC}"
        echo -e "$PR_BODY"
    fi

    echo -e "${GREEN}完了しました！${NC}"
    echo -e "コミットメッセージ:\n${YELLOW}$COMMIT_MESSAGE${NC}"
    echo -e "${GREEN}プルリクエストが作成されました。レビュー後にマージしてください。${NC}"
    echo -e "${BLUE}注意: このPRに他のコミットを追加する場合は、コミットメッセージの規約に従ってください。${NC}"
    echo -e "${GREEN}マージされると、GitHub Actionsが自動的にバージョン管理とリリースを行います。${NC}"
else
    # コミットのみの場合
    echo -e "${GREEN}コミットのみを行います...${NC}"
    git add -A
    git commit -m "$COMMIT_MESSAGE"
    echo -e "${GREEN}コミットが完了しました。${NC}"
fi 
