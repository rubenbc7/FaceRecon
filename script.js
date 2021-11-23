const imageUpload = document.getElementById('imageUpload');

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssMobilenetc1.loadFromUri('/models')
]).then(start)

async function start(){
    const container = document.createElement('div')
    container.style.position = 'relative'
    document.body.append(container)
    const labeledFaceDecriptors = await loadLabeledImages()
    const faceMatcher = faceapi.FaceMatcher(labeledFaceDecriptors, 0.6)
    let image
    let canvas
    document.body.append('Loaded')

    imageUpload.addEventListener('change', async () =>{
        if (image) image.remove()
        if (canvas) canvas.remove()

        image = await faceapi.bufferToImage(imageUpload.file[0])
        container.append(image)
        canvas.append(canvas)
        canvas = faceapi.createCanvasFromMedia(image)
        container.append(canvas)

        const displaySize = {width: image.width, height: image.height}

        faceapi.matchDimensions(canvas, displaySize)

        const detections = await faceapi.detectAllFaces(image).widthFaceLandMarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))

        results.forEach((results, i) =>{
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, {label: results.toString() })
            drawBox.draw(canvas)
        })
    })
}

function loadLabeledImages() {
    const labels = ['Black Widow','Captain America', 'Captain Marvel', 'Hawkeye', 'Jime Rhodes', 'Thor', 'Tony Stark']
    return Promise.all(
        labels.map(async label =>{
            const descriptions = []
            for (let i =1; i <= 2; i++){
                const img = await faceapi.fetchImage(`https://mawe.mx/face/iamges/${label}/${i}.jgp`)
                const detections = await faceapi.detectionSingleFace(img).withFaceLandmarks().withFaceDescriptors()
                descriptions.push(detections.descriptor)
            }

            return new faceapi.labeledFaceDecriptors(label, descriptions) 
        })
    )
}