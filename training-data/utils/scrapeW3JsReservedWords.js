const puppeteer = require("puppeteer");

(async () => {
  const browser = await puppeteer.launch();

  const page = await browser.newPage();
  await page.goto("https://www.w3schools.com/js/js_reserved.asp");

  const code = await page.evaluate(() =>
    document
      .querySelector(".w3-table-all")
      .textContent.split("\n")
      .filter(z => z !== "")
      .map(s => s.replace("*", ""))
      .join("\n")
  );

  console.log(code);

  await browser.close();
})();
