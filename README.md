
# Manual do Site - Pietro S. Consonni

Este projeto foi convertido para um **Site Estático (HTML Puro)**.
Isso significa que ele é super leve, não precisa de instalação complexa (Jekyll/Ruby/Docker) e não precisa de "build". O que você vê nos arquivos `.html` é exatamente o que aparece no site.

## 1. Como Rodar no seu Computador (Localhost)

Para visualizar o site na sua máquina antes de publicar:

1.  Abra o terminal na pasta do projeto.
2.  Rode o servidor simples do Python:
    ```powershell
    python -m http.server 4000
    ```
3.  Acesse no navegador: [http://localhost:4000](http://localhost:4000)

## 2. Estrutura das Pastas

Tudo o que importa está organizado assim:

*   **`index.html`**: A página inicial (Home).
*   **`pages/`**: Onde estão todas as outras páginas do site (Research, Teaching, Prices, etc.).
*   **`files/data/`**: Onde ficam os arquivos CSV para os gráficos.
*   **`assets/`**: Arquivos técnicos (CSS, Javascript, fontes) - *Evite mexer aqui a menos que queira mudar o design.*

## 3. Como Editar o Conteúdo das Páginas

Cada página do menu é um arquivo `.html` dentro da pasta `pages`. Para alterar o texto:

1.  Abra o arquivo desejado no seu editor de código (VS Code, Bloco de Notas, etc).
2.  Procure o texto que deseja mudar e edite.
3.  **Cuidado:** Mantenha as "tags" (coisas entre `< >`) intactas. Mude apenas o texto preto.

**Onde está cada página?**
*   **Research:** `pages/research.html`
*   **Teaching:** `pages/teaching.html`
*   **Prices:** `pages/prices.html`
*   **Price Analyses:** `pages/price-analyses.html`
*   **Macro Analyses:** `pages/macro-analyses.html`
*   **About / CV:** `pages/about.html` e `pages/cv.html`

Exemplo: Para mudar um título na página de Research, abra `pages/research.html`, dê um Ctrl+F pelo título antigo e escreva o novo.

## 4. Como Atualizar Gráficos (Página "Teste Espaco")

A página `pages/teste.html` gera gráficos automaticamente a partir de arquivos CSV.
Para adicionar um novo gráfico:

1.  **Salve o arquivo:**
    Coloque seu arquivo CSV (ex: `novo_feijao.csv`) na pasta `files/data/`.

2.  **Registre o arquivo:**
    O site não consegue "adivinhar" quais arquivos estão na pasta (por segurança dos navegadores). Você precisa avisar que ele existe.
    *   Abra o arquivo `pages/teste.html`.
    *   Procure por esta linha (lá pelo começo, linha ~239):
        ```javascript
        const FALLBACK_CSV = ["acúcar.csv", "brl_usd.csv", ... ];
        ```
    *   Adicione o nome do seu arquivo na lista, entre aspas:
        ```javascript
        const FALLBACK_CSV = ["novo_feijao.csv", "acúcar.csv", ... ];
        ```

3.  **Pronto!**
    Atualize a página e o gráfico aparecerá automaticamente.

## 5. Como Publicar (Colocar no Ar)

Para atualizar o site oficial (GitHub Pages):

1.  Salve todos os arquivos que você editou.
2.  No terminal, digite:
    ```bash
    git add .
    git commit -m "Atualizando conteúdo do site"
    git push origin master
    ```
3.  Aguarde alguns segundos e o site online será atualizado.
