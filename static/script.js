
    function handleKeyPress(e) {
        let key = e.keyCode || e.which;
        if (key === 13) {
            findSimilarMusic();
        }
    }

    function findSimilarMusic() {
        post('/find_similar_music', function (rawResponse) {
            let i = 1
            console.log(rawResponse)
            let response = JSON.parse(rawResponse)
            let finalHtml = '<tr align="center" valign="center"><th></th><th>Track name</th><th>Similarity score</th></tr>'
            response.forEach((trackData) => {
                finalHtml += createTrackRow(i++, trackData['neighbour'], trackData['distance']['value'])
            })
            document.getElementById('result').innerHTML = finalHtml
            document.getElementById('trackNameText').innerHTML = response[0]['origin']
        })
    }

    function createTrackRow(i, trackName, similarityScore) {
        return '<tr><td><p style="margin-right: 10px">' + i + "." + '</p>' +
               '</td><td><p style="margin-right: 10px">' + trackName + '</p>' +
               '</td><td align="center" valign="center">' + (similarityScore * 100).toFixed(2) + '%</td></tr>'
    }

    function uploadTrack() {
        post('/find_similar_music2', function (rawResponse) {
            console.log('sent')
        })
    }

    function post(url, callback) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function () {
            callback(xmlHttp.responseText)
        }
        xmlHttp.open("POST", url, true); // true for asynchronous
        xmlHttp.send(null);
    }