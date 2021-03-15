---
title: "部落格搬家記"
date: 2021-02-16T17:28:58+08:00
draft: false
weight: 2

description: "紀錄一下Deploy Hugo Site和設定Github Action遇到的坑"
categories: ["front-end", "diary"]
series: []
tags: ["front-end", "Hugo", "blog", "diary", "Utterance"]

keywords:
- Hugo
- Github Action
- Front-End

cover:
    image: "img/just_imgs/taipei3.jpg"
    relative: false # To use relative path for cover image, used in hugo Page-bundles
---

因為寫DL筆記時會用到大量數學符號，就索性把原先在Github上的[DL_DB_Quick_Notes](https://github.com/FrankCCCCC/DL_DB_Quick_Note)搬過來了，配合LATEX寫筆記順手很多，原先的Repo應該只會剩下收集Paper用。而最近生活上有些轉折，也許也會順便放些隨筆雜記，但就依心情而定。

目前用的主題是[PaperMod](https://github.com/adityatelange/hugo-PaperMod)，整體設計算令人滿意，只不過在Deploy Hugo遇到蠻多麻煩，這邊簡單記錄一下

## 設定Github Page Action

參考[PaperMod ExampleSite的gh-pages.yml設定](https://github.com/adityatelange/hugo-PaperMod/blob/exampleSite/.github/workflows/gh-pages.yml)，自己再作一些修改，大致如下

```
name: Build GH-Pages

on:
  push:
    paths-ignore:
      - 'images/**'
      - 'LICENSE'
      - 'README.md'
    branches:
      - master
  workflow_dispatch:
    # manual run

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Git checkout
        uses: actions/checkout@v2
        with:
          ref: master

      - name: Get Theme
        run: git submodule update --init --recursive

      - name: Update theme to Latest commit
        run: git submodule update --remote --merge

      - name: Setup hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: 'latest'

      - name: Build
        run: hugo --buildDrafts --gc --verbose --minify

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.YOUR_TOKEN }}
          publish_dir: ./public
```

`name`隨意填上該Action的名稱，`on`則是event trigger，指定在甚麼時候trigger `jobs`，並在`branch`內設定要被build的分支，

`step`的話就是填寫Build到Deploy的步驟，可以大致看出依序是先git checkout到要被build的分支，然後下載更新themes，然後設定Hugo，再來用指令`hugo --buildDrafts --gc --verbose --minify` build static website，其中`--buildDrafts`代表把草稿也build好公開，`--minify`則是盡量壓縮build好的網站大小。

最後Deploy步驟則是要把build好的網頁推到`gh-pages` branch上，注意分支`gh-pages`需要手動先創好推上Github才能讓Github Action自動Build，否則會報錯。同時Github會需要全限修改分支內容，要去Settings->Developer settings->Personal access tokens裡面新增token，並給予workflow、admin:repo_hook權限(我有給這些權限，但是不確定那些真的會需要)，按確定後把Token複製下來。

![](/blog/img/move_blog/personal_access_tokens.png)
*感謝[Milk Midi](https://milkmidi.medium.com/%E6%B7%B1%E5%85%A5%E4%BD%86%E4%B8%8D%E6%B7%BA%E5%87%BA-%E5%A6%82%E4%BD%95%E7%94%A8-github-actions-%E8%87%AA%E5%8B%95%E7%99%BC%E4%BD%88-gh-pages-8183464dfe84)整理*

接下來再到your_blog_repo->Settings->Secrets新增Actions secrets，把Token貼上，再把Actions secrets 的名字貼到`YOUR_TOKEN`就好。

![](/blog/img/move_blog/secrets.png)
*感謝[Milk Midi](https://milkmidi.medium.com/%E6%B7%B1%E5%85%A5%E4%BD%86%E4%B8%8D%E6%B7%BA%E5%87%BA-%E5%A6%82%E4%BD%95%E7%94%A8-github-actions-%E8%87%AA%E5%8B%95%E7%99%BC%E4%BD%88-gh-pages-8183464dfe84)整理*

比較詳細的圖解說明可以參考[這篇](https://milkmidi.medium.com/%E6%B7%B1%E5%85%A5%E4%BD%86%E4%B8%8D%E6%B7%BA%E5%87%BA-%E5%A6%82%E4%BD%95%E7%94%A8-github-actions-%E8%87%AA%E5%8B%95%E7%99%BC%E4%BD%88-gh-pages-8183464dfe84)，太懶得紀錄這種瑣碎操作。

## Latex 設定
參考[這篇](https://geoffruddock.com/math-typesetting-in-hugo/)

### Step 1

首先在安裝好的主題裡面`layouts/partials/mathjax_support.html`新增`.html`檔

```
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$','$$'], ['\\[', '\\]']],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }
  };

  window.addEventListener('load', (event) => {
      document.querySelectorAll("mjx-container").forEach(function(x){
        x.parentElement.classList += 'has-jax'})
    });

</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

### Step 2

在`layouts/partials/header.html`的`</head>` tag裡面再新增這段code

```
{{ partial "mathjax_support.html" . }}
```
### Step 3

最後在`assets/css/header.css`檔裡面再加上這段code，如果沒有這個檔案，就把code加到所有頁面都會用到的CSS檔

```
code.has-jax {
    -webkit-font-smoothing: antialiased;
    background: inherit !important;
    border: none !important;
    font-size: 100%;
}
```

以上，完工。給個範例

```
$$a_{PI}(x|D) = E[u(x) | x, D] = \int_{-\infty}^{f'} \mathcal{N}(f; \mu(x), \kappa(x, x)) \ df
=\phi(f'; \mu(x), \kappa(x, x))$$
```

顯示很完美

$$a_{PI}(x|D) = E[u(x) | x, D] = \int_{-\infty}^{f'} \mathcal{N}(f; \mu(x), \kappa(x, x)) \ df
=\phi(f'; \mu(x), \kappa(x, x))$$

只不過會Mathjax在parse底線的時，有時候會有一點問題，如

```
$\begin{equation} x_t = \mathop{\arg\max}_{x \in X} \ \ a_{PI}(x|D_{1:t−1}) \end{equation}$
```

顯示會出現

$\begin{equation} x_t = \mathop{\arg\max}_{x \in X} \ \ a_{PI}(x|D_{1:t−1}) \end{equation}$

會壞掉，解決辦法就是前後都加個 **`** 符號，變成

```
`$\begin{equation} x_t = \mathop{\arg\max}_{x \in X} \ \ a_{PI}(x|D_{1:t−1}) \end{equation}$`
```

`$\begin{equation} x_t = \mathop{\arg\max}_{x \in X} \ \ a_{PI}(x|D_{1:t−1}) \end{equation}$`

顯示就會正常了，但是會以Inline Code的方式顯示，就會變的小一點。這種現象的主要原因是在Step 1我們是把LATEX Code和Markdown的code一起parse，但Markdown語法本身就會用到底線，這會導致重複定義同一個符號，所以就需要而外把LATEX抓出來塞到Inline Code裡面處理，就不會重複定義。但基本上很少遇到有問題的情況，若遇到顯示有問題再加 **`** 就好。

## MD 圖片路徑設定

由於Hugo在parse圖片連結時並不會對圖片連結進行轉換，也就是說遇到`/blog/img/figure.jpg`會預設用絕對路徑解析，會抓到到`baseurl/blog/img/figure.jpg`的圖片，遇到`img/figure.jpg`則會預設用相對路徑解析，就會抓到`current_url/blog/img/figure.jpg`的圖片。

但麻煩的是Deploy到Github Page上後，預設網址為`https://{user_account}.github.io/{repository_name}/`，網站的絕對路徑前綴就會變成`https://{user_account}.github.io`，而非Hugo config裡面設定的baseurl，會導致圖片完全無法顯示。

解決辦法就是[把所有MD的圖片路徑都轉成完整的網址](https://discourse.gohugo.io/t/image-path/1721)，即在config.yml加上下面這行code就解決了

```
canonifyURLs: true
```

## MD 圖片路徑設定 Vol.2
用了一段時間後，Post數量超過一頁能顯示的範圍，才發現Paginaiton有問題，原因就在於`canonifyURLs: true`會把Pagination的URL都弄成絕對路徑，把網址搞壞。解決方法就是把canonifyURL設回預設值`canonifyURLs: false`，但這樣就會把md圖片又搞壞。

最後最後的解決辦法就是把md的圖片路徑前面加上`/blog`，因為部屬在Github Page上的Reop名是blog，所以Github Page的網址就是`https://{Username}.github.io/blog`，，而Hugo是靜態網頁產生器，Build後的圖片都會serve在`/img`下，所以就直接用專案路徑加Build好的圖片路徑來寫md(改來改去超煩的)。

## 安裝Utterances留言板

[Utterances Official](https://utteranc.es/)

相較於GitTalk, GitMent等基於Github Issue的留言插件，Utterances要求的權限最少，因此決定採用。設定安裝可參考[此篇](https://www.dazhuanlan.com/2019/12/05/5de8934e6f081/)

## 線上壓縮圖片
Google推出的[Squoosh](https://squoosh.app/)挺方便的。

# Finally
最後的成果就如這個Blog，Repo的連結則在[這裡](https://github.com/FrankCCCCC/blog)，有興趣的人應該可以參考一下code。雖然說這種文章應該多少對有需要的人有所幫助，而且很容易就可以獲得成就感(因為寫這種文章不需要動腦)，相對於人生，實在容易許多。

最後工商一下[舊部落格 DL DB Quick Note](https://github.com/FrankCCCCC/DL_DB_Quick_Note)，但文章應該會慢慢整理搬過來，對RL, NTK, 費茲傑羅, 黃麗群和獨立音樂有興趣的也歡迎討論，只不過我可能只回重要事項，其他看心情回覆就是了。

# Reference

- [Render LaTeX math expressions in Hugo with MathJax 3](https://geoffruddock.com/math-typesetting-in-hugo/)
- [深入但不淺出，如何用 github actions 自動發佈 gh-pages](https://milkmidi.medium.com/%E6%B7%B1%E5%85%A5%E4%BD%86%E4%B8%8D%E6%B7%BA%E5%87%BA-%E5%A6%82%E4%BD%95%E7%94%A8-github-actions-%E8%87%AA%E5%8B%95%E7%99%BC%E4%BD%88-gh-pages-8183464dfe84)
- [使用Hugo+Github Pages建置Blog](https://www.jianshu.com/p/58c644011f7d)
- [git submodule 教學](https://medium.com/@kmsh3ng/git-submodule-%E6%95%99%E5%AD%B8-96ab0255c88c)
- [Hugo Theme: adityatelange/hugo-PaperMod](https://github.com/adityatelange/hugo-PaperMod)
- [List of LaTeX mathematical symbols](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)