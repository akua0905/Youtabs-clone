import * as ort from "onnxruntime-web";
import {
  BasicPitch,
  outputToNotesPoly,
  addPitchBendsToNoteEvents,
  noteFramesToTime
} from "@spotify/basic-pitch";
import { Midi } from "@tonejs/midi";
import JSZip from "jszip";


// ============================================================
// 設定
// ============================================================

// UVR-MDX-NET-Inst_HQ_3
// 約66.8MB
const MDX_MODEL =
  "https://huggingface.co/masszhou/mdxnet/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx";

const BASIC_PITCH_MODEL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";


// MDX-Net
const SAMPLE_RATE = 44100;

const MDX_N_FFT = 6144;
const MDX_HOP = 1024;
const MDX_DIM_F = 3072;
const MDX_DIM_T = 256;

const MDX_CHUNK_SIZE =
  MDX_HOP * (MDX_DIM_T - 1);

const MDX_OVERLAP = 2;

const MDX_COMPENSATION = 1.022;


// Basic Pitch
const BASIC_PITCH_SAMPLE_RATE = 22050;


// MIDI
const MIDI_BPM = 120;

const MERGE_GAP = 0.08;
const MIN_NOTE_DURATION = 0.08;
const MAX_FILL_GAP = 0.05;


// ============================================================
// DOM
// ============================================================

const audioInput =
  document.getElementById("audio");

const convertButton =
  document.getElementById("convert");

const statusElement =
  document.getElementById("status");

const downloadElement =
  document.getElementById("download");

const progressBar =
  document.getElementById("progressBar");


// ============================================================
// 状態
// ============================================================

let selectedFile = null;


// ============================================================
// UI
// ============================================================

function setStatus(message) {
  statusElement.textContent = message;
}


function setProgress(value) {

  const percent =
    Math.max(
      0,
      Math.min(
        100,
        value * 100
      )
    );

  progressBar.style.width =
    `${percent}%`;
}


function waitForBrowser() {

  return new Promise(
    resolve => {
      setTimeout(
        resolve,
        0
      );
    }
  );
}


// ============================================================
// ファイル選択
// ============================================================

audioInput.addEventListener(
  "change",
  () => {

    selectedFile =
      audioInput.files?.[0] ?? null;

    downloadElement.style.display =
      "none";

    setProgress(0);

    if (!selectedFile) {

      setStatus(
        "音源を選択してください。"
      );

      return;
    }

    setStatus(
      `選択済み\n${selectedFile.name}`
    );

  }
);


// ============================================================
// メイン処理
// ============================================================

convertButton.addEventListener(
  "click",
  async () => {

    if (!selectedFile) {

      setStatus(
        "先に音源を選択してください。"
      );

      return;
    }


    convertButton.disabled = true;

    downloadElement.style.display =
      "none";


    let mdxSession = null;


    try {

      // ======================================================
      // 0. WebGPU確認
      // ======================================================

      setStatus(
        "音源分離AIを確認しています……"
      );

      setProgress(0.01);


      const webgpuAvailable =
        await isWebGPUAvailable();


      if (!webgpuAvailable) {

        throw new Error(
          "この端末ではWebGPUを利用できません。\n\n" +
          "iPhone SafariではWebGPU対応状況によって、" +
          "MDX-Netのブラウザ推論を実行できない場合があります。"
        );

      }


      // ======================================================
      // 1. 音源読み込み
      // ======================================================

      setStatus(
        "音源を読み込んでいます……"
      );

      setProgress(0.03);


      const fileBuffer =
        await selectedFile.arrayBuffer();


      const decodedAudio =
        await decodeAudio(
          fileBuffer
        );


      // ======================================================
      // 2. 44.1kHz Stereo
      // ======================================================

      setStatus(
        "音源を44.1kHzステレオに変換しています……"
      );

      setProgress(0.06);


      const stereo =
        await convertToStereo44100(
          decodedAudio
        );


      const originalLeft =
        stereo.left;

      const originalRight =
        stereo.right;


      // decoded AudioBufferは不要
      // ここからはFloat32Arrayだけで処理する。


      // ======================================================
      // 3. MDX-Net読み込み
      // ======================================================

      setStatus(
        "音源分離AIを読み込んでいます……\n" +
        "初回は約67MBのモデルを読み込みます。"
      );

      setProgress(0.09);


      mdxSession =
        await createMDXSession();


      // ======================================================
      // 4. ボーカル・伴奏分離
      // ======================================================

      setStatus(
        "ボーカル／伴奏を分離しています……"
      );

      setProgress(0.12);


      const separated =
        await separateWithMDX(
          mdxSession,
          originalLeft,
          originalRight,
          progress => {

            setProgress(
              0.12 +
              progress * 0.45
            );

            setStatus(
              "ボーカル／伴奏を分離しています……\n" +
              `${Math.round(progress * 100)}%`
            );

          }
        );


      const vocals =
        separated.vocals;

      const instrumental =
        separated.instrumental;


      // ======================================================
      // MDX-Net解放
      // ======================================================

      try {

        if (
          mdxSession &&
          typeof mdxSession.release === "function"
        ) {

          await mdxSession.release();

        }

      } catch {}

      mdxSession = null;


      // ======================================================
      // 5. Basic Pitch
      // ======================================================

      setStatus(
        "Basic Pitchを準備しています……"
      );

      setProgress(0.60);


      const basicPitch =
        new BasicPitch(
          BASIC_PITCH_MODEL
        );


      // ======================================================
      // 6. ボーカル → MIDI
      // ======================================================

      setStatus(
        "ボーカルをMIDIに変換しています……"
      );

      setProgress(0.62);


      const vocalsMidi =
        await audioToMidi(
          basicPitch,
          vocals,
          progress => {

            setProgress(
              0.62 +
              progress * 0.15
            );

          }
        );


      // ======================================================
      // 7. 伴奏 → MIDI
      // ======================================================

      setStatus(
        "伴奏をMIDIに変換しています……"
      );

      setProgress(0.78);


      const instrumentalMidi =
        await audioToMidi(
          basicPitch,
          instrumental,
          progress => {

            setProgress(
              0.78 +
              progress * 0.15
            );

          }
        );


      // ======================================================
      // 8. ZIP
      // ======================================================

      setStatus(
        "ZIPファイルを生成しています……"
      );

      setProgress(0.95);


      const zip =
        new JSZip();


      zip.file(
        "vocals.mid",
        vocalsMidi
      );


      zip.file(
        "instrumental.mid",
        instrumentalMidi
      );


      const zipBlob =
        await zip.generateAsync({
          type: "blob",
          compression: "DEFLATE"
        });


      const url =
        URL.createObjectURL(
          zipBlob
        );


      const baseName =
        selectedFile.name
          .replace(
            /\.[^/.]+$/,
            ""
          );


      downloadElement.href =
        url;

      downloadElement.download =
        `${baseName}_採譜.zip`;

      downloadElement.textContent =
        "ZIPを保存";

      downloadElement.style.display =
        "block";


      setProgress(1);


      setStatus(
        "解析完了\n\n" +
        "vocals.mid\n" +
        "instrumental.mid\n\n" +
        "ZIPにまとめました。"
      );


    } catch (error) {

      console.error(
        error
      );


      const message =
        error instanceof Error
          ? `${error.name}: ${error.message}`
          : String(error);


      setStatus(
        "エラーが発生しました。\n\n" +
        message
      );


      setProgress(0);


    } finally {

      if (mdxSession) {

        try {

          if (
            typeof mdxSession.release === "function"
          ) {

            await mdxSession.release();

          }

        } catch {}

      }

      convertButton.disabled =
        false;

    }

  }
);


// ============================================================
// WebGPU
// ============================================================

async function isWebGPUAvailable() {

  if (
    !("gpu" in navigator)
  ) {

    return false;

  }


  try {

    const adapter =
      await navigator.gpu.requestAdapter();

    return !!adapter;

  } catch {

    return false;

  }

}


// ============================================================
// Audio Decode
// ============================================================

async function decodeAudio(
  arrayBuffer
) {

  const context =
    new AudioContext();


  try {

    return await context.decodeAudioData(
      arrayBuffer.slice(0)
    );

  } finally {

    try {

      await context.close();

    } catch {}

  }

}


// ============================================================
// 44.1kHz Stereo
// ============================================================

async function convertToStereo44100(
  source
) {

  const targetRate =
    SAMPLE_RATE;


  const targetLength =
    Math.ceil(
      source.duration *
      targetRate
    );


  const offline =
    new OfflineAudioContext(
      2,
      targetLength,
      targetRate
    );


  const sourceNode =
    offline.createBufferSource();


  sourceNode.buffer =
    source;


  sourceNode.connect(
    offline.destination
  );


  sourceNode.start(0);


  const rendered =
    await offline.startRendering();


  const left =
    rendered
      .getChannelData(0)
      .slice();


  let right;


  if (
    rendered.numberOfChannels >= 2
  ) {

    right =
      rendered
        .getChannelData(1)
        .slice();

  } else {

    right =
      left.slice();

  }


  return {
    left,
    right
  };

}


// ============================================================
// MDX-Net Session
// ============================================================

async function createMDXSession() {

  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = 1;

  ort.env.wasm.proxy = false;


  const session =
    await ort.InferenceSession.create(
      MDX_MODEL,
      {
        executionProviders: [
          "webgpu"
        ],

        graphOptimizationLevel:
          "all",

        enableCpuMemArena:
          false,

        enableMemPattern:
          false
      }
    );


  console.log(
    "MDX input names:",
    session.inputNames
  );


  console.log(
    "MDX output names:",
    session.outputNames
  );


  return session;

}


// ============================================================
// MDX-Net Separation
// ============================================================

async function separateWithMDX(

  session,

  left,

  right,

  onProgress

) {

  const total =
    left.length;


  const instrumentalLeft =
    new Float32Array(
      total
    );

  const instrumentalRight =
    new Float32Array(
      total
    );


  const weight =
    new Float32Array(
      total
    );


  const chunkSize =
    MDX_CHUNK_SIZE;


  const overlapSize =
    Math.floor(
      chunkSize /
      MDX_OVERLAP
    );


  const stride =
    chunkSize -
    overlapSize;


  const chunkCount =
    Math.max(
      1,
      Math.ceil(
        Math.max(
          0,
          total -
          chunkSize
        ) /
        stride
      ) + 1
    );


  for (
    let chunkIndex = 0;
    chunkIndex < chunkCount;
    chunkIndex++
  ) {

    const start =
      chunkIndex *
      stride;


    const end =
      Math.min(
        start +
        chunkSize,
        total
      );


    const chunkLength =
      end -
      start;


    const paddedLeft =
      new Float32Array(
        chunkSize
      );

    const paddedRight =
      new Float32Array(
        chunkSize
      );


    paddedLeft.set(
      left.subarray(
        start,
        end
      )
    );


    paddedRight.set(
      right.subarray(
        start,
        end
      )
    );


    // 最後のチャンクを自然に埋める
    if (
      chunkLength <
      chunkSize
    ) {

      fillReflection(
        paddedLeft,
        chunkLength
      );

      fillReflection(
        paddedRight,
        chunkLength
      );

    }


    // ========================================================
    // STFT
    // ========================================================

    const spectrum =
      createMDXSpectrum(
        paddedLeft,
        paddedRight
      );


    const inputName =
      session.inputNames[0];

    const outputName =
      session.outputNames[0];


    const tensor =
      new ort.Tensor(
        "float32",
        spectrum,
        [
          1,
          4,
          MDX_DIM_F,
          MDX_DIM_T
        ]
      );


    const result =
      await session.run({
        [inputName]:
          tensor
      });


    const output =
      result[outputName];


    if (!output) {

      throw new Error(
        "MDX-Netの出力が取得できませんでした。\n" +
        `出力: ${session.outputNames.join(", ")}`
      );

    }


    // ========================================================
    // iSTFT
    // ========================================================

    const stem =
      mdxSpectrumToAudio(
        output.data
      );


    const window =
      makeCrossfadeWindow(
        chunkSize,
        overlapSize
      );


    for (
      let i = 0;
      i < chunkLength;
      i++
    ) {

      const index =
        start +
        i;


      if (
        index >= total
      ) {

        break;

      }


      const w =
        window[i];


      instrumentalLeft[index] +=
        stem.left[i] *
        w *
        MDX_COMPENSATION;


      instrumentalRight[index] +=
        stem.right[i] *
        w *
        MDX_COMPENSATION;


      weight[index] +=
        w;

    }


    onProgress(
      (chunkIndex + 1) /
      chunkCount
    );


    await waitForBrowser();

  }


  // ========================================================
  // Overlap normalization
  // ========================================================

  for (
    let i = 0;
    i < total;
    i++
  ) {

    const w =
      Math.max(
        weight[i],
        1e-8
      );


    instrumentalLeft[i] /=
      w;


    instrumentalRight[i] /=
      w;

  }


  // ========================================================
  // Vocals = Original - Instrumental
  // ========================================================

  const vocalsLeft =
    new Float32Array(
      total
    );

  const vocalsRight =
    new Float32Array(
      total
    );


  for (
    let i = 0;
    i < total;
    i++
  ) {

    vocalsLeft[i] =
      left[i] -
      instrumentalLeft[i];


    vocalsRight[i] =
      right[i] -
      instrumentalRight[i];

  }


  return {

    vocals: {
      left:
        vocalsLeft,

      right:
        vocalsRight
    },

    instrumental: {
      left:
        instrumentalLeft,

      right:
        instrumentalRight
    }

  };

}


// ============================================================
// MDX STFT
// ============================================================

function createMDXSpectrum(
  left,
  right
) {

  const output =
    new Float32Array(
      4 *
      MDX_DIM_F *
      MDX_DIM_T
    );


  const fftSize =
    MDX_N_FFT;


  const half =
    MDX_DIM_F;


  const frameCount =
    MDX_DIM_T;


  const hop =
    MDX_HOP;


  const window =
    makeHannWindow(
      fftSize
    );


  const leftPadded =
    reflectPad(
      left,
      fftSize / 2
    );


  const rightPadded =
    reflectPad(
      right,
      fftSize / 2
    );


  const real =
    new Float64Array(
      fftSize
    );

  const imag =
    new Float64Array(
      fftSize
    );


  for (
    let frame = 0;
    frame < frameCount;
    frame++
  ) {

    const offset =
      frame *
      hop;


    // ========================================================
    // Left
    // ========================================================

    for (
      let i = 0;
      i < fftSize;
      i++
    ) {

      real[i] =
        leftPadded[
          offset + i
        ] *
        window[i];

      imag[i] = 0;

    }


    fft(
      real,
      imag,
      false
    );


    for (
      let bin = 0;
      bin < half;
      bin++
    ) {

      const index =
        (
          0 *
          half +
          bin
        ) *
        frameCount +
        frame;


      output[index] =
        real[bin];

      output[
        half *
        frameCount +
        index
      ] =
        imag[bin];

    }


    // ========================================================
    // Right
    // ========================================================

    for (
      let i = 0;
      i < fftSize;
      i++
    ) {

      real[i] =
        rightPadded[
          offset + i
        ] *
        window[i];

      imag[i] = 0;

    }


    fft(
      real,
      imag,
      false
    );


    const rightRealOffset =
      2 *
      half *
      frameCount;


    const rightImagOffset =
      3 *
      half *
      frameCount;


    for (
      let bin = 0;
      bin < half;
      bin++
    ) {

      output[
        rightRealOffset +
        bin *
        frameCount +
        frame
      ] =
        real[bin];


      output[
        rightImagOffset +
        bin *
        frameCount +
        frame
      ] =
        imag[bin];

    }

  }


  return output;

}


// ============================================================
// MDX iSTFT
// ============================================================

function mdxSpectrumToAudio(
  data
) {

  const fftSize =
    MDX_N_FFT;


  const half =
    MDX_DIM_F;


  const frameCount =
    MDX_DIM_T;


  const hop =
    MDX_HOP;


  const chunkSize =
    MDX_CHUNK_SIZE;


  const paddedLength =
    chunkSize +
    fftSize;


  const left =
    new Float32Array(
      paddedLength
    );

  const right =
    new Float32Array(
      paddedLength
    );


  const normalization =
    new Float64Array(
      paddedLength
    );


  const window =
    makeHannWindow(
      fftSize
    );


  const real =
    new Float64Array(
      fftSize
    );

  const imag =
    new Float64Array(
      fftSize
    );


  const leftImagOffset =
    half *
    frameCount;


  const rightRealOffset =
    2 *
    half *
    frameCount;


  const rightImagOffset =
    3 *
    half *
    frameCount;


  for (
    let frame = 0;
    frame < frameCount;
    frame++
  ) {

    // ========================================================
    // Left
    // ========================================================

    for (
      let bin = 0;
      bin < half;
      bin++
    ) {

      const index =
        bin *
        frameCount +
        frame;


      real[bin] =
        data[index];

      imag[bin] =
        data[
          leftImagOffset +
          index
        ];

    }


    // Reconstruct Nyquist
    real[half] = 0;
    imag[half] = 0;


    // Conjugate half
    for (
      let bin = 1;
      bin < half;
      bin++
    ) {

      real[
        fftSize - bin
      ] =
        real[bin];

      imag[
        fftSize - bin
      ] =
        -imag[bin];

    }


    fft(
      real,
      imag,
      true
    );


    const offset =
      frame *
      hop;


    for (
      let i = 0;
      i < fftSize;
      i++
    ) {

      const w =
        window[i];


      left[
        offset + i
      ] +=
        real[i] *
        w;


      normalization[
        offset + i
      ] +=
        w *
        w;

    }


    // ========================================================
    // Right
    // ========================================================

    for (
      let bin = 0;
      bin < half;
      bin++
    ) {

      const index =
        bin *
        frameCount +
        frame;


      real[bin] =
        data[
          rightRealOffset +
          index
        ];


      imag[bin] =
        data[
          rightImagOffset +
          index
        ];

    }


    real[half] = 0;
    imag[half] = 0;


    for (
      let bin = 1;
      bin < half;
      bin++
    ) {

      real[
        fftSize - bin
      ] =
        real[bin];

      imag[
        fftSize - bin
      ] =
        -imag[bin];

    }


    fft(
      real,
      imag,
      true
    );


    for (
      let i = 0;
      i < fftSize;
      i++
    ) {

      const w =
        window[i];


      right[
        offset + i
      ] +=
        real[i] *
        w;

    }

  }


  // ==========================================================
  // Normalize and remove center padding
  // ==========================================================

  const resultLeft =
    new Float32Array(
      chunkSize
    );

  const resultRight =
    new Float32Array(
      chunkSize
    );


  const trim =
    fftSize / 2;


  for (
    let i = 0;
    i < chunkSize;
    i++
  ) {

    const source =
      i +
      trim;


    const norm =
      Math.max(
        normalization[source],
        1e-8
      );


    resultLeft[i] =
      left[source] /
      norm;


    resultRight[i] =
      right[source] /
      norm;

  }


  return {

    left:
      resultLeft,

    right:
      resultRight

  };

}


// ============================================================
// Reflection padding
// ============================================================

function reflectPad(
  input,
  pad
) {

  const p =
    Math.floor(pad);


  const output =
    new Float32Array(
      input.length +
      p * 2
    );


  for (
    let i = 0;
    i < p;
    i++
  ) {

    const source =
      Math.min(
        input.length - 1,
        p - i
      );

    output[i] =
      input[source];

  }


  output.set(
    input,
    p
  );


  for (
    let i = 0;
    i < p;
    i++
  ) {

    const source =
      Math.max(
        0,
        input.length -
        2 -
        i
      );

    output[
      p +
      input.length +
      i
    ] =
      input[source];

  }


  return output;

}


// ============================================================
// Reflection fill
// ============================================================

function fillReflection(
  buffer,
  validLength
) {

  if (
    validLength <= 0
  ) {

    return;

  }


  for (
    let i = validLength;
    i < buffer.length;
    i++
  ) {

    const offset =
      i -
      validLength;


    const source =
      Math.max(
        0,
        validLength -
        2 -
        (
          offset %
          Math.max(
            1,
            validLength - 1
          )
        )
      );


    buffer[i] =
      buffer[source];

  }

}


// ============================================================
// Hann
// ============================================================

function makeHannWindow(
  size
) {

  const window =
    new Float64Array(
      size
    );


  for (
    let i = 0;
    i < size;
    i++
  ) {

    window[i] =
      0.5 *
      (
        1 -
        Math.cos(
          2 *
          Math.PI *
          i /
          size
        )
      );

  }


  return window;

}


// ============================================================
// Crossfade
// ============================================================

function makeCrossfadeWindow(
  length,
  overlap
) {

  const window =
    new Float32Array(
      length
    );


  window.fill(1);


  for (
    let i = 0;
    i < overlap;
    i++
  ) {

    const x =
      i /
      overlap;


    window[i] =
      0.5 -
      0.5 *
      Math.cos(
        Math.PI *
        x
      );


    window[
      length -
      1 -
      i
    ] =
      window[i];

  }


  return window;

}


// ============================================================
// FFT
// ============================================================

function fft(
  real,
  imag,
  inverse
) {

  const n =
    real.length;


  // Bit reversal
  for (
    let i = 1,
    j = 0;
    i < n;
    i++
  ) {

    let bit =
      n >> 1;


    for (
      ;
      j & bit;
      bit >>= 1
    ) {

      j ^=
        bit;

    }


    j ^=
      bit;


    if (
      i < j
    ) {

      let temp =
        real[i];

      real[i] =
        real[j];

      real[j] =
        temp;


      temp =
        imag[i];

      imag[i] =
        imag[j];

      imag[j] =
        temp;

    }

  }


  for (
    let len = 2;
    len <= n;
    len <<= 1
  ) {

    const angle =
      (
        inverse
          ? 2
          : -2
      ) *
      Math.PI /
      len;


    const wLenReal =
      Math.cos(angle);


    const wLenImag =
      Math.sin(angle);


    for (
      let i = 0;
      i < n;
      i += len
    ) {

      let wReal = 1;
      let wImag = 0;


      const halfLen =
        len >> 1;


      for (
        let j = 0;
        j < halfLen;
        j++
      ) {

        const uReal =
          real[
            i + j
          ];

        const uImag =
          imag[
            i + j
          ];


        const vIndex =
          i +
          j +
          halfLen;


        const vReal =
          real[vIndex] *
          wReal -
          imag[vIndex] *
          wImag;


        const vImag =
          real[vIndex] *
          wImag +
          imag[vIndex] *
          wReal;


        real[
          i + j
        ] =
          uReal +
          vReal;


        imag[
          i + j
        ] =
          uImag +
          vImag;


        real[vIndex] =
          uReal -
          vReal;


        imag[vIndex] =
          uImag -
          vImag;


        const nextWReal =
          wReal *
          wLenReal -
          wImag *
          wLenImag;


        wImag =
          wReal *
          wLenImag +
          wImag *
          wLenReal;


        wReal =
          nextWReal;

      }

    }

  }


  if (inverse) {

    for (
      let i = 0;
      i < n;
      i++
    ) {

      real[i] /=
        n;

      imag[i] /=
        n;

    }

  }

}


// ============================================================
// Audio → Basic Pitch → MIDI
// ============================================================

async function audioToMidi(

  basicPitch,

  stereo,

  onProgress

) {

  const audioBuffer =
    await createBasicPitchAudioBuffer(
      stereo
    );


  const frames = [];
  const onsets = [];
  const contours = [];


  await basicPitch.evaluateModel(

    audioBuffer,

    (
      frameBatch,
      onsetBatch,
      contourBatch
    ) => {

      frames.push(
        ...frameBatch
      );

      onsets.push(
        ...onsetBatch
      );

      contours.push(
        ...contourBatch
      );

    },

    progress => {

      onProgress(
        progress
      );

    }

  );


  let notes =
    outputToNotesPoly(
      frames,
      onsets,
      0.25,
      0.25,
      5
    );


  notes =
    addPitchBendsToNoteEvents(
      contours,
      notes
    );


  let timedNotes =
    noteFramesToTime(
      notes
    );


  timedNotes =
    normalizeNoteObjects(
      timedNotes
    );


  timedNotes =
    mergeSamePitchNotes(
      timedNotes
    );


  timedNotes =
    removeShortNotes(
      timedNotes
    );


  timedNotes =
    fillSmallGaps(
      timedNotes
    );


  timedNotes.sort(
    (
      a,
      b
    ) =>
      a.startTimeSeconds -
      b.startTimeSeconds
  );


  const midi =
    new Midi();


  midi.header.setTempo(
    MIDI_BPM
  );


  const track =
    midi.addTrack();


  for (
    const note
    of timedNotes
  ) {

    const pitch =
      Math.round(
        note.pitchMidi
      );


    const time =
      Math.max(
        0,
        note.startTimeSeconds
      );


    const duration =
      Math.max(
        0.01,
        note.durationSeconds
      );


    const velocity =
      Math.max(
        0.01,

        Math.min(
          1,
          Number(
            note.amplitude ?? 0.8
          )
        )
      );


    track.addNote({

      midi:
        pitch,

      time:
        time,

      duration:
        duration,

      velocity:
        velocity

    });

  }


  return midi.toArray();

}


// ============================================================
// Basic Pitch AudioBuffer
// ============================================================

async function createBasicPitchAudioBuffer(
  stereo
) {

  const sourceRate =
    SAMPLE_RATE;


  const targetRate =
    BASIC_PITCH_SAMPLE_RATE;


  const length =
    stereo.left.length;


  const mono =
    new Float32Array(
      length
    );


  for (
    let i = 0;
    i < length;
    i++
  ) {

    mono[i] =
      (
        stereo.left[i] +
        stereo.right[i]
      ) *
      0.5;

  }


  const sourceBuffer =
    new AudioBuffer({
      length:
        length,

      numberOfChannels:
        1,

      sampleRate:
        sourceRate
    });


  sourceBuffer
    .getChannelData(0)
    .set(mono);


  const targetLength =
    Math.ceil(
      sourceBuffer.duration *
      targetRate
    );


  const offline =
    new OfflineAudioContext(
      1,
      targetLength,
      targetRate
    );


  const source =
    offline.createBufferSource();


  source.buffer =
    sourceBuffer;


  source.connect(
    offline.destination
  );


  source.start(0);


  return await offline.startRendering();

}


// ============================================================
// MIDI Note Normalization
// ============================================================

function normalizeNoteObjects(
  notes
) {

  return notes.map(
    note => {

      const start =
        Number(
          note.startTimeSeconds ??
          note.startTime ??
          0
        );


      const duration =
        Number(
          note.durationSeconds ??
          note.duration ??
          0
        );


      const pitch =
        Number(
          note.pitchMidi ??
          note.pitch ??
          0
        );


      const amplitude =
        Number(
          note.amplitude ??
          0.8
        );


      return {

        ...note,

        startTimeSeconds:
          start,

        durationSeconds:
          duration,

        pitchMidi:
          pitch,

        amplitude:
          amplitude

      };

    }
  );

}


// ============================================================
// 同音ノート結合
// ============================================================

function mergeSamePitchNotes(
  notes
) {

  if (
    notes.length <= 1
  ) {

    return notes;

  }


  const sorted =
    [...notes].sort(
      (
        a,
        b
      ) =>
        a.startTimeSeconds -
        b.startTimeSeconds
    );


  const merged = [];


  for (
    const note
    of sorted
  ) {

    const previous =
      merged[
        merged.length - 1
      ];


    if (!previous) {

      merged.push({
        ...note
      });

      continue;

    }


    const previousEnd =
      previous.startTimeSeconds +
      previous.durationSeconds;


    const noteEnd =
      note.startTimeSeconds +
      note.durationSeconds;


    const gap =
      note.startTimeSeconds -
      previousEnd;


    const samePitch =
      Math.round(
        previous.pitchMidi
      ) ===
      Math.round(
        note.pitchMidi
      );


    if (
      samePitch &&
      gap >= -0.03 &&
      gap <= MERGE_GAP
    ) {

      previous.durationSeconds =
        Math.max(
          previousEnd,
          noteEnd
        ) -
        previous.startTimeSeconds;


      previous.amplitude =
        Math.max(
          previous.amplitude ?? 0,
          note.amplitude ?? 0
        );

    } else {

      merged.push({
        ...note
      });

    }

  }


  return merged;

}


// ============================================================
// 短すぎるノート削除
// ============================================================

function removeShortNotes(
  notes
) {

  return notes.filter(
    note =>
      note.durationSeconds >=
      MIN_NOTE_DURATION
  );

}


// ============================================================
// 小さな隙間を埋める
// ============================================================

function fillSmallGaps(
  notes
) {

  const sorted =
    [...notes].sort(
      (
        a,
        b
      ) =>
        a.startTimeSeconds -
        b.startTimeSeconds
    );


  for (
    let i = 0;
    i < sorted.length - 1;
    i++
  ) {

    const current =
      sorted[i];

    const next =
      sorted[i + 1];


    if (
      Math.round(
        current.pitchMidi
      ) !==
      Math.round(
        next.pitchMidi
      )
    ) {

      continue;

    }


    const currentEnd =
      current.startTimeSeconds +
      current.durationSeconds;


    const gap =
      next.startTimeSeconds -
      currentEnd;


    if (
      gap > 0 &&
      gap <= MAX_FILL_GAP
    ) {

      current.durationSeconds +=
        gap;

    }

  }


  return sorted;

}