//express
module.exports = {

server: function Server(){
    const path = require('path');
    const express = require('express');
    const app = express();

    const dir = path.join(__dirname, './');

    app.use(express.static(dir));

    app.listen(8080, function () {
        console.log('Listening on http://localhost:8080/');
    });

    let mime = {
    html: 'text/html',
    txt: 'text/plain',
    css: 'text/css',
    gif: 'image/gif',
    jpg: 'image/jpeg',
    png: 'image/png',
    svg: 'image/svg+xml',
    js: 'application/javascript'
    };

    app.get('*', function (req, res) {
        let file = path.join(dir, req.path.replace(/\/$/, 'index.html'));
    if (file.indexOf(dir + path.sep) !== 0) {
        return res.status(403).end('Forbidden');
    }
    let type = mime[path.extname(file).slice(1)] || 'text/plain';
    let s = fs.createReadStream(file);
    s.on('open', function () {
        res.set('Content-Type', type);
        s.pipe(res);
    });
    s.on('error', function () {
        res.set('Content-Type', 'text/plain');
        res.status(404).end('Not found');
    });
    });
}
}
