document.addEventListener('DOMContentLoaded', function () {
    onPageLoaded()
});

function findSimilarMusic() {
    post('/find_similar_music', function (rawResponse) {
        let i = 1
        let response = JSON.parse(rawResponse)
        let finalHtml = createHeader()
        response.forEach((trackData) => {
            finalHtml += createTrackRow(i++, trackData['neighbour'], trackData['distance']['value'])
        })
        finalHtml += '</div>'
        document.getElementById('result').innerHTML = finalHtml
        console.log(finalHtml)
        document.getElementById('trackNameText').display = true
        document.getElementById('trackNameText').innerHTML = response[0]['origin']
    })
}

function createHeader() {
    return '<div class="list-item-horizontal" style="background-color: var(--accent-color)">' +
        '<p style="flex: 1;">Название трека</p>' +
        '<p>Процент схожести</p>' +
        '</div>'
}

function createTrackRow(i, trackName, similarityScore) {
    return '<div class="list-item-horizontal"><p>' + i + "." + '</p>' +
        '<p>' + trackName + '</p>' +
        '<p>' + (similarityScore * 100).toFixed(2) + '%</p></div>'
}

function post(url, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        callback(xmlHttp.responseText)
    }
    xmlHttp.open("POST", url, true); // true for asynchronous
    xmlHttp.send(null);
}

function onPageLoaded() {
    setUpCollapsibleListsForTracks()
}

function setUpCollapsibleListsForTracks() {
    const trackItemElements = document.getElementsByClassName("list-item-horizontal");
    let i;

    for (i = 0; i < trackItemElements.length; i++) {
        console.log('loaded')
        trackItemElements[i].addEventListener("click", function () {
            this.classList.toggle("active");
            let content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        });
    }
}
