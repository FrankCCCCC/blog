---
title: "部落格搬家記"
date: 2021-02-16T17:28:58+08:00
draft: false

categories: ["front-end", "diary"]
series: []
tags: ["front-end", "hugo", "blog", "diary", "Utterance"]

cover:
    image: "img/just_imgs/city_station.jpg"
    relative: false # To use relative path for cover image, used in hugo Page-bundles
---

因為寫DL筆記時會用到大量數學符號，就索性把原先在Github上的[DL_DB_Quick_Notes](https://github.com/FrankCCCCC/DL_DB_Quick_Note)搬過來了，配合LATEX寫筆記順手很多，原先的Repo應該只會剩下收集Paper用。而最近生活上有些轉折，也許也會順便放些隨筆雜記，但就依心情而定。

目前用的主題是[PaperMod](https://github.com/adityatelange/hugo-PaperMod)，整體設計算令人滿意，只不過在Deploy HUGO遇到蠻多麻煩，這邊簡單記錄一下

## 設定Github Page Action

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
$$a_{PI}(x|D) = \mathbb{E}[u(x) | x, D] = \int_{-\infty}^{f'} \mathcal{N}(f; \mu(x), \kappa(x, x)) \ df
=\phi(f'; \mu(x), \kappa(x, x))$$
```

顯示很完美

$$a_{PI}(x|D) = \mathbb{E}[u(x) | x, D] = \int_{-\infty}^{f'} \mathcal{N}(f; \mu(x), \kappa(x, x)) \ df
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

由於HUGO在parse圖片連結時並不會對圖片連結進行轉換，也就是說遇到`/img/figure.jpg`會預設用絕對路徑解析，會抓到到baseurl/img/figure.jpg的圖片，遇到`img/figure.jpg`則會預設用相對路徑解析，就會抓到current_url/img/figure.jpg的圖片。

但麻煩的是Deploy到Github Page上後，預設網址為https://{user_account}.github.io/{repository_name}/，網站的絕對路徑前綴就會變成https://{user_account}.github.io，而非HUGO config裡面設定的baseurl，會導致圖片完全無法顯示。

解決辦法就是[把所有MD的圖片路徑都轉成完整的網址](https://discourse.gohugo.io/t/image-path/1721)，即在config.yml加上下面這行code就解決了

```
canonifyURLs: true
```

## 安裝Utterances留言板

[Utterances Official](https://utteranc.es/)

相較於GitTalk, GitMent等基於Github Issue的留言插件，Utterances要求的權限最少，因此決定採用。設定安裝可參考[此篇](https://www.dazhuanlan.com/2019/12/05/5de8934e6f081/)


# Reference

- [Render LaTeX math expressions in Hugo with MathJax 3](https://geoffruddock.com/math-typesetting-in-hugo/)
- [深入但不淺出，如何用 github actions 自動發佈 gh-pages](https://milkmidi.medium.com/%E6%B7%B1%E5%85%A5%E4%BD%86%E4%B8%8D%E6%B7%BA%E5%87%BA-%E5%A6%82%E4%BD%95%E7%94%A8-github-actions-%E8%87%AA%E5%8B%95%E7%99%BC%E4%BD%88-gh-pages-8183464dfe84)
- [使用Hugo+Github Pages建置Blog](https://www.jianshu.com/p/58c644011f7d)
- [git submodule 教學](https://medium.com/@kmsh3ng/git-submodule-%E6%95%99%E5%AD%B8-96ab0255c88c)
- [HUGO Theme: adityatelange/hugo-PaperMod](https://github.com/adityatelange/hugo-PaperMod)