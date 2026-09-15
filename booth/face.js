// Talosian face layer: biometric IDENTITY, on-device, in the browser.
//
// This recognizes WHO someone appears to be (their face). It is deliberately
// NOT the authority layer. Continuity crypto decides whether a command may act;
// a recognized face never authorizes anything on its own. The whole point Talosian
// makes: even a confirmed face (or a held-up photo, or a look-alike) is identity,
// not authority. So this module only ever produces "who does the camera think
// this is", which the console displays and the agent references, while the gate
// still requires a real continuity proof to act.
//
// Real, offline: face-api.js (@vladmandic) + local models in face/models. No
// cloud, no upload, the webcam frame never leaves the browser. Face descriptors
// (128-float vectors) are the only thing stored, per enrolled commander.

const MODEL_URL = './face/models';
let ready = false;
let loading = null;

// Lazy-load the library + models once (on first camera use), so the page opens
// instantly and only pays the ~7MB model cost if face is actually used.
export async function initFace() {
  if (ready) return true;
  if (loading) return loading;
  loading = (async () => {
    if (!window.faceapi) {
      await loadScript('./face/face-api.js');
    }
    const f = window.faceapi;
    await f.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await f.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await f.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    ready = true;
    return true;
  })();
  return loading;
}

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res; s.onerror = () => rej(new Error('face-api load failed'));
    document.head.appendChild(s);
  });
}

// Attach the webcam to a <video> element. Returns the stream (stop it to release
// the camera). Never records or uploads, the stream stays in the page.
export async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 320, height: 240, facingMode: 'user' }, audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();
  return stream;
}

// Compute the 128-d descriptor for the single most prominent face in a video/
// image element, or null if no face is found. This vector IS the face identity.
export async function faceDescriptor(el) {
  if (!ready) return null;
  const f = window.faceapi;
  const det = await f
    .detectSingleFace(el, new f.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 }))
    .withFaceLandmarks()
    .withFaceDescriptor();
  return det ? Array.from(det.descriptor) : null;
}

// Match a live descriptor against enrolled commanders. Returns
// { id, name, distance } for the best match under the threshold, or null.
// Lower distance = more similar; ~0.5 is a good same-person threshold.
export function matchFace(descriptor, enrolled, threshold = 0.52) {
  if (!descriptor || !enrolled.length) return null;
  let best = null;
  for (const c of enrolled) {
    if (!c.faceDescriptor) continue;
    const d = euclidean(descriptor, c.faceDescriptor);
    if (best === null || d < best.distance) best = { id: c.id, name: c.name, distance: d };
  }
  return best && best.distance <= threshold ? best : null;
}

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}
