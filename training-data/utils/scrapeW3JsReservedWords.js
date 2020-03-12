const Nightmare = require('nightmare')
const nightmare = Nightmare({ show: false })

nightmare
  .goto('https://www.w3schools.com/js/js_reserved.asp')
  .wait('.w3-table-all')
  .evaluate(() => document
    .querySelector('.w3-table-all')
    .textContent
    .split('\n')
    .filter(z => z !== '')
    .map(s => s.replace('*', ''))
    .join('\n')
  )
  .end()
  .then(console.log)
  .catch(error => {
    console.error('Search failed:', error)
  })