const puppeteer = require("puppeteer");

async function asyncForEach(array, callback) {
  for (let index = 0; index < array.length; index++) {
    await callback(array[index], index, array);
  }
}

(async () => {
  const browser = await puppeteer.launch();

  const page = await browser.newPage();
  await page.goto(
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference"
  );

  const referenceLinks = await page.evaluate(() =>
    [...document.querySelectorAll("#wikiArticle li a")].map(({ href }) => href)
  );

  // console.log(JSON.stringify(referenceLinks))

  const codeBlocks = [];

  await asyncForEach(referenceLinks, async link => {
    await page.goto(link);
    const codeBlock = await page.evaluate(() =>
      [...document.querySelectorAll("code.language-js")].map(c => c.innerText)
    );
    codeBlocks.push([page.url(), ...codeBlock]);
  });

  await Promise.all(codeBlocks).then(res => console.log(JSON.stringify(res)));
  await browser.close();
})();
