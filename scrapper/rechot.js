import {JSDOM}  from 'jsdom';
import jQuery   from 'jquery';
import low      from 'lowdb';
import FileSync from 'lowdb/adapters/FileSync';
 
const adapter = new FileSync('./out/rechot.json')
const db = low(adapter);
 
// Set some defaults (required if your JSON file is empty)
db.defaults({ jokes: [] })
  .write()
 
const getLinksAsync = async page => {
  return await JSDOM.fromURL(`http://www.rechot.com/dowcipy,${page}.html`)
    .then(dom => {
      const window = dom.window;
      const document = window.document;
      const $ = jQuery(window);
 
      var jokes = db.get('jokes');
 
      $('.dowcip > div:not(:last-child)').not('.n').each(function(){
        var innerText = $(this).text()
        //console.log( innerText );
        var re = /Rechot nr: (?<rechotId>\d+) - kategoria: (?<cat>.+)autor: [^\n]+\s+(?<joke>[\s\S]+)\s(?<likes>\d+)\s+(?<dislikes>\d+)Skomentuj/gm;
        var result = re.exec(innerText);
 
        jokes = jokes.push(result.groups)
      });
      jokes.write();
    }
  );
};
 
(async () => {
  for (let i = 341; i>0; i--) {
    console.log(`Fetching page ${i}`);
    await getLinksAsync(i);
  }
})()
