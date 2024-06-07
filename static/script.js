document.addEventListener('DOMContentLoaded', function () {
    onPageUpdated()
});

function findSimilarMusic() {
    post('/find_similar_music', function (rawResponse) {
        let i = 1
        let response = JSON.parse(rawResponse)
        let finalHtml = createHeader()
        console.log(response)
        response['neighbour_data_list'].forEach((trackData) => {
            finalHtml += createTrackRow(i++, trackData['neighbour'], trackData['total_duration'], trackData['similarity']['value'])
            finalHtml += createDropDownList(trackData['neighbour'], response['segment_data_list'])
        })
        finalHtml += '</div>'
        document.getElementById('result').innerHTML = finalHtml
        // console.log(finalHtml)
        document.getElementById('trackNameText').display = true
        document.getElementById('trackNameText').innerHTML = response['original_track_name'] + ' (' + response['original_track_total_duration'] + ')'
        onPageUpdated()
    })
}

function createHeader() {
    return '<div class="list-item-horizontal" style="background-color: var(--accent-color); margin-bottom: 16px">' +
        '<p style="flex: 1;">Название аудиозаписи</p>' +
        '<p>Процент схожести</p>' +
        '</div>'
}

function createTrackRow(i, trackName, duration, similarityScore) {
    return '<div class="list-item-horizontal"><p>' + i + "." + '</p>' +
        '<p>' + trackName + '</p>' +
        '<p class="secondary">(' + duration + ')</p>' +
        '<p>' + similarityScore + '%</p></div>'
}

function createDropDownList(currentTrackName, segmentSimilarityMap) {
    let html = '<div class="similarity-container">'
    let i = 1
    segmentSimilarityMap[currentTrackName].forEach((segment) => {
        html += '<div class="similarity-item">' +
            '<p>Секция ' + i++ + ' </p>' +
            '<p class="secondary">(' + segment['duration_start'] + ' - ' + segment['duration_end'] + ') </p>' +
            '<p>Совпадение </p>' +
            '<p>' + segment['similarity']['value'] + '%</p></div>'
    })
    return html + '</div>'
}

function post(url, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function () {
        callback(xmlHttp.responseText)
    }
    xmlHttp.open("POST", url, true); // true for asynchronous
    xmlHttp.send(null);
}

function onPageUpdated() {
    setUpCollapsibleListsForTracks()
}

function setUpCollapsibleListsForTracks() {
    const trackItemElements = document.getElementsByClassName("list-item-horizontal");
    let i;

    for (i = 0; i < trackItemElements.length; i++) {
        trackItemElements[i].addEventListener("click", function () {
            console.log('click')
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
