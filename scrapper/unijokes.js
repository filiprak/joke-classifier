import {JSDOM}  from 'jsdom';
import low      from 'lowdb';
import FileSync from 'lowdb/adapters/FileSync';
 
const adapter = new FileSync('./out/unijokes.json')
const db = low(adapter);
 
// Set some defaults (required if your JSON file is empty)
db.defaults({ jokes: [] })
  .write()
 
const getLinksAsync = async (page) => {
  return await JSDOM.fromURL(`https://unijokes.com/${page}/`)
    .then(dom => {
      const window = dom.window;
      const document = window.document;
 
      var jokes = db.get('jokes');
 
      [...document.querySelectorAll('.j')].forEach( el => {
        let panel = el.querySelector('.panel');
        jokes = jokes.push({
          joke: el.firstChild.textContent,
          unijokesId: document.querySelector("[data-id]").dataset.id,
          ratingValue: panel.querySelector("[itemprop=ratingValue]").textContent,
          ratingCount: panel.querySelector("[itemprop=ratingCount]").textContent,
          categories: [...panel.querySelectorAll("a")]
            .filter((x,i) => i > 0)
            .map(el => el.text)
        })
      });
 
      jokes.write();
    }
  );
};
 
(async () => {
  for (let i = 1; i<1402; i++) {
    console.log(`Fetching page ${i}`);
    await getLinksAsync(i);
  }
})()
