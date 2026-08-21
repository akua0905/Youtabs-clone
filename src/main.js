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
// Model
// ============================================================

const MDX_MODEL =
  "https://huggingface.co/masszhou/mdxnet/resolve/main/UVR-MDX-NET-Inst_HQ_3.onnx";

const BASIC_PITCH_MODEL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";


// ============================================================
// MDX-Net configuration
// ============================================================

const SAMPLE_RATE = 44100;

const N_FFT = 6144;
const HOP = 1024;

const DIM_F = 3072;
const DIM_T = 8;

const CHUNK_SIZE =
  HOP * (DIM_T - 1);

const TRIM =
  N_FFT / 2;

const GEN_SIZE =
  CHUNK_SIZE - 2 * TRIM;

const OVERLAP = 2;

const COMPENSATION = 1.022;


// ============================================================
// Basic Pitch
// ============================================================

const BASIC_PITCH_SAMPLE_RATE = 22050;


// ============================================================
// MIDI
// ============================================================

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
// State
// ============================================================

let selectedFile = null;


// ============================================================
// UI
// ============================================================

function setStatus(message) {
  if (statusElement) {
    statusElement.textContent = message;
  }
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

  if (progressBar) {
    progressBar.style.width =
      `${percent}%`;
  }
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
// File
// ============================================================

if (audioInput) {

  audioInput.addEventListener(
    "change",
    () => {

      selectedFile =
        audioInput.files?.[0] ?? null;

      if (downloadElement) {
        downloadElement.style.display =
          "none";
      }

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

}


// ============================================================
// Main
// ============================================================

if (convertButton) {

  convertButton.addEventListener(
    "click",
    async () => {

      if (!selectedFile) {

        setStatus(
          "先に音源を選択してください。"
        );

        return;
      }


      convertButton.disabled =
        true;


      if (downloadElement) {
        downloadElement.style.display =
          "none";
      }


      let session = null;


      try {

        // ------------------------------------------------------
        // 1. WebGPU
        // ------------------------------------------------------

        setStatus(
          "WebGPUを確認しています……"
        );

        setProgress(0.01);


        if (
          !("gpu" in navigator)
        ) {

          throw new Error(
            "このブラウザではWebGPUを利用できません。\n\n" +
            "iPhone SafariではWebGPU対応版のSafariを使用してください。"
          );

        }


        const adapter =
          await navigator.gpu.requestAdapter();


        if (!adapter) {

          throw new Error(
            "WebGPUアダプターを取得できませんでした。"
          );

        }


        await waitForBrowser();


        // ------------------------------------------------------
        // 2. Decode
        // ------------------------------------------------------

        setStatus(
          "音源を読み込んでいます……"
        );

        setProgress(0.03);


        const fileBuffer =
          await selectedFile.arrayBuffer();


        const decoded =
          await decodeAudio(
            fileBuffer
          );


        await waitForBrowser();


        // ------------------------------------------------------
        // 3. Stereo 44.1kHz
        // ------------------------------------------------------

        setStatus(
          "音源を44.1kHzステレオに変換しています……"
        );

        setProgress(0.06);


        const stereo =
          await convertToStereo44100(
            decoded
          );


        const originalLeft =
          stereo.left;

        const originalRight =
          stereo.right;


        // decoded AudioBufferを解放しやすくする
        decoded.getChannelData(0).fill(0);


        await waitForBrowser();


        // ------------------------------------------------------
        // 4. MDX session
        // ------------------------------------------------------

        setStatus(
          "音源分離AIを読み込んでいます……\n" +
          "初回は約67MBのモデルを読み込みます。"
        );

        setProgress(0.08);


        session =
          await createMDXSession();


        await waitForBrowser();


        // ------------------------------------------------------
        // 5. Separation
        // ------------------------------------------------------

        setStatus(
          "ボーカル／伴奏を分離しています……"
        );

        setProgress(0.10);


        const separated =
          await separateWithMDX(
            session,
            originalLeft,
            originalRight,
            progress => {

              setProgress(
                0.10 +
                progress * 0.47
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


        // ------------------------------------------------------
        // 6. Release MDX
        // ------------------------------------------------------

        try {

          if (
            session &&
            typeof session.release ===
              "function"
          ) {

            await session.release();

          }

        } catch {}

        session = null;


        await waitForBrowser();


        // ------------------------------------------------------
        // 7. Basic Pitch
        // ------------------------------------------------------

        setStatus(
          "Basic Pitchを準備しています……"
        );

        setProgress(0.59);


        const basicPitch =
          new BasicPitch(
            BASIC_PITCH_MODEL
          );


        await waitForBrowser();


        // ------------------------------------------------------
        // 8. Vocals MIDI
        // ------------------------------------------------------

        setStatus(
          "ボーカルをMIDIに変換しています……"
        );

        setProgress(0.61);


        const vocalsMidi =
          await audioToMidi(
            basicPitch,
            vocals,
            progress => {

              setProgress(
                0.61 +
                progress * 0.17
              );

            }
          );


        await waitForBrowser();


        // ------------------------------------------------------
        // 9. Instrumental MIDI
        // ------------------------------------------------------

        setStatus(
          "伴奏をMIDIに変換しています……"
        );

        setProgress(0.79);


        const instrumentalMidi =
          await audioToMidi(
            basicPitch,
            instrumental,
            progress => {

              setProgress(
                0.79 +
                progress * 0.15
              );

            }
          );


        await waitForBrowser();


        // ------------------------------------------------------
        // 10. ZIP
        // ------------------------------------------------------

        setStatus(
          "ZIPファイルを生成しています……"
        );

        setProgress(0.96);


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
          selectedFile.name.replace(
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

        if (session) {

          try {

            if (
              typeof session.release ===
                "function"
            ) {

              await session.release();

            }

          } catch {}

        }


        convertButton.disabled =
          false;

      }

    }
  );

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
// Convert to 44.1kHz Stereo
// ============================================================

async function convertToStereo44100(
  source
) {

  const targetLength =
    Math.ceil(
      source.duration *
      SAMPLE_RATE
    );


  const offline =
    new OfflineAudioContext(
      2,
      targetLength,
      SAMPLE_RATE
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
// MDX Session
// ============================================================

async function createMDXSession() {

  ort.env.wasm.simd =
    true;

  ort.env.wasm.numThreads =
    1;

  ort.env.wasm.proxy =
    false;


  if (
    ort.env.webgpu
  ) {

    ort.env.webgpu.powerPreference =
      "low-power";

  }


  const session =
    await ort.InferenceSession.create(
      MDX_MODEL,
      {

        executionProviders: [
          "webgpu"
        ],

        graphOptimizationLevel:
          "basic",

        enableCpuMemArena:
          false,

        enableMemPattern:
          false

      }
    );


  console.log(
    "MDX inputs:",
    session.inputNames
  );


  console.log(
    "MDX outputs:",
    session.outputNames
  );


  return session;

}


// ============================================================
// MDX Separation
// ============================================================

async function separateWithMDX(

  session,

  originalLeft,

  originalRight,

  onProgress

) {

  const total =
    originalLeft.length;


  /*
   * UVR MDX-Net:
   *
   * dim_f = 3072
   * dim_t = 8
   * n_fft = 6144
   * hop = 1024
   *
   * chunk_size =
   * 1024 * (8 - 1)
   * = 7168 samples
   */


  const outputLeft =
    new Float32Array(
      total
    );

  const outputRight =
    new Float32Array(
      total
    );


  const weight =
    new Float32Array(
      total
    );


  const hopSize =
    Math.floor(
      CHUNK_SIZE /
      OVERLAP
    );


  const genSize =
    GEN_SIZE;


  const paddedLength =
    total +
    2 * TRIM;


  const padRemainder =
    (
      genSize -
      (
        paddedLength -
        CHUNK_SIZE
      ) %
      genSize
    ) %
    genSize;


  const finalLength =
    paddedLength +
    padRemainder;


  const paddedLeft =
    new Float32Array(
      finalLength
    );

  const paddedRight =
    new Float32Array(
      finalLength
    );


  paddedLeft.set(
    originalLeft,
    TRIM
  );

  paddedRight.set(
    originalRight,
    TRIM
  );


  const chunkCount =
    Math.max(
      1,
      Math.ceil(
        (
          finalLength -
          CHUNK_SIZE
        ) /
        hopSize
      ) + 1
    );


  for (
    let chunkIndex = 0;
    chunkIndex < chunkCount;
    chunkIndex++
  ) {

    const sourceStart =
      chunkIndex *
      hopSize;


    const chunkLeft =
      new Float32Array(
        CHUNK_SIZE
      );

    const chunkRight =
      new Float32Array(
        CHUNK_SIZE
      );


    const available =
      Math.min(
        CHUNK_SIZE,
        finalLength -
        sourceStart
      );


    if (
      available > 0
    ) {

      chunkLeft.set(
        paddedLeft.subarray(
          sourceStart,
          sourceStart +
          available
        )
      );


      chunkRight.set(
        paddedRight.subarray(
          sourceStart,
          sourceStart +
          available
        )
      );

    }


    setStatus(
      "音源分離AIを実行しています……\n" +
      `チャンク ${chunkIndex + 1}/${chunkCount}`
    );


    await waitForBrowser();


    // --------------------------------------------------------
    // STFT
    // --------------------------------------------------------

    const spectrum =
      createMDXSpectrum(
        chunkLeft,
        chunkRight
      );


    await waitForBrowser();


    const inputName =
      session.inputNames[0];


    const outputName =
      session.outputNames[0];


    /*
     * IMPORTANT:
     *
     * [1, 4, 3072, 8]
     *
     * 前のコードの [1,4,3072,256] は誤り。
     */

    const tensor =
      new ort.Tensor(
        "float32",
        spectrum,
        [
          1,
          4,
          DIM_F,
          DIM_T
        ]
      );


    // spectrumへの参照を保持するのはtensorだけ
    // 推論後にまとめて解放する。


    const result =
      await session.run(
        {
          [inputName]:
            tensor
        }
      );


    const output =
      result[outputName];


    if (!output) {

      throw new Error(
        "MDX-Netの出力が見つかりません。\n" +
        `出力名: ${session.outputNames.join(", ")}`
      );

    }


    const outputData =
      output.data;


    const stem =
      mdxSpectrumToAudio(
        outputData
      );


    // --------------------------------------------------------
    // Overlap Add
    // --------------------------------------------------------

    const blend =
      createBlendWindow(
        CHUNK_SIZE,
        hopSize
      );


    /*
     * モデル出力の中央部分だけ使用する。
     *
     * MDX-NetのSTFTはcenter=True相当なので、
     * 左右TRIMを捨てる。
     */

    const generatedLength =
      Math.min(
        GEN_SIZE,
        stem.left.length -
        2 * TRIM
      );


    for (
      let i = 0;
      i < generatedLength;
      i++
    ) {

      const sourceIndex =
        i +
        TRIM;


      const destinationIndex =
        sourceStart +
        i +
        TRIM;


      if (
        destinationIndex >=
        finalLength
      ) {

        break;

      }


      const w =
        blend[
          i %
          blend.length
        ];


      outputLeft[
        Math.max(
          0,
          destinationIndex -
          TRIM
        )
      ] +=
        stem.left[
          sourceIndex
        ] *
        COMPENSATION *
        w;


      outputRight[
        Math.max(
          0,
          destinationIndex -
          TRIM
        )
      ] +=
        stem.right[
          sourceIndex
        ] *
        COMPENSATION *
        w;

    }


    /*
     * weight
     *
     * ここでは簡易的にチャンクの有効範囲を
     * 正規化する。
     */

    const weightStart =
      Math.max(
        0,
        sourceStart
      );


    const weightEnd =
      Math.min(
        total,
        sourceStart +
        CHUNK_SIZE
      );


    for (
      let i = weightStart;
      i < weightEnd;
      i++
    ) {

      weight[i] += 1;

    }


    /*
     * Safariに処理を返す。
     *
     * これが重要。
     * 長時間JSを占有し続けない。
     */

    onProgress(
      (chunkIndex + 1) /
      chunkCount
    );


    await waitForBrowser();

  }


  // ----------------------------------------------------------
  // Normalize
  // ----------------------------------------------------------

  for (
    let i = 0;
    i < total;
    i++
  ) {

    const w =
      Math.max(
        weight[i],
        1
      );


    outputLeft[i] /=
      w;


    outputRight[i] /=
      w;

  }


  /*
   * instrumental
   */

  const instrumentalLeft =
    outputLeft.slice(
      0,
      total
    );

  const instrumentalRight =
    outputRight.slice(
      0,
      total
    );


  /*
   * vocals = original - instrumental
   */

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
      originalLeft[i] -
      instrumentalLeft[i];


    vocalsRight[i] =
      originalRight[i] -
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
// STFT
// ============================================================

function createMDXSpectrum(
  left,
  right
) {

  const result =
    new Float32Array(
      4 *
      DIM_F *
      DIM_T
    );


  const window =
    makeHannWindow(
      N_FFT
    );


  /*
   * center=True
   *
   * 3072 samplesを左右にpadding。
   */

  const paddedLeft =
    new Float32Array(
      CHUNK_SIZE +
      N_FFT
    );

  const paddedRight =
    new Float32Array(
      CHUNK_SIZE +
      N_FFT
    );


  paddedLeft.set(
    left,
    TRIM
  );

  paddedRight.set(
    right,
    TRIM
  );


  /*
   * 6144-point FFT
   *
   * 6144 = 3 × 2048
   */

  const real =
    new Float32Array(
      N_FFT
    );

  const imag =
    new Float32Array(
      N_FFT
    );


  for (
    let frame = 0;
    frame < DIM_T;
    frame++
  ) {

    const offset =
      frame *
      HOP;


    // --------------------------------------------------------
    // Left
    // --------------------------------------------------------

    real.fill(0);
    imag.fill(0);


    for (
      let i = 0;
      i < N_FFT;
      i++
    ) {

      real[i] =
        paddedLeft[
          offset + i
        ] *
        window[i];

    }


    fft6144(
      real,
      imag,
      false
    );


    for (
      let bin = 0;
      bin < DIM_F;
      bin++
    ) {

      const index =
        bin *
        DIM_T +
        frame;


      result[index] =
        real[bin];


      result[
        DIM_F *
        DIM_T +
        index
      ] =
        imag[bin];

    }


    // --------------------------------------------------------
    // Right
    // --------------------------------------------------------

    real.fill(0);
    imag.fill(0);


    for (
      let i = 0;
      i < N_FFT;
      i++
    ) {

      real[i] =
        paddedRight[
          offset + i
        ] *
        window[i];

    }


    fft6144(
      real,
      imag,
      false
    );


    const rightRealOffset =
      2 *
      DIM_F *
      DIM_T;


    const rightImagOffset =
      3 *
      DIM_F *
      DIM_T;


    for (
      let bin = 0;
      bin < DIM_F;
      bin++
    ) {

      const index =
        bin *
        DIM_T +
        frame;


      result[
        rightRealOffset +
        index
      ] =
        real[bin];


      result[
        rightImagOffset +
        index
      ] =
        imag[bin];

    }

  }


  return result;

}


// ============================================================
// iSTFT
// ============================================================

function mdxSpectrumToAudio(
  data
) {

  const paddedLength =
    CHUNK_SIZE +
    N_FFT;


  const left =
    new Float32Array(
      paddedLength
    );

  const right =
    new Float32Array(
      paddedLength
    );


  const norm =
    new Float32Array(
      paddedLength
    );


  const window =
    makeHannWindow(
      N_FFT
    );


  const real =
    new Float32Array(
      N_FFT
    );

  const imag =
    new Float32Array(
      N_FFT
    );


  const leftImagOffset =
    DIM_F *
    DIM_T;


  const rightRealOffset =
    2 *
    DIM_F *
    DIM_T;


  const rightImagOffset =
    3 *
    DIM_F *
    DIM_T;


  for (
    let frame = 0;
    frame < DIM_T;
    frame++
  ) {

    // --------------------------------------------------------
    // Left
    // --------------------------------------------------------

    real.fill(0);
    imag.fill(0);


    for (
      let bin = 0;
      bin < DIM_F;
      bin++
    ) {

      const index =
        bin *
        DIM_T +
        frame;


      real[bin] =
        data[index];


      imag[bin] =
        data[
          leftImagOffset +
          index
        ];

    }


    reconstructConjugateSpectrum(
      real,
      imag
    );


    fft6144(
      real,
      imag,
      true
    );


    const offset =
      frame *
      HOP;


    for (
      let i = 0;
      i < N_FFT;
      i++
    ) {

      const w =
        window[i];


      left[
        offset + i
      ] +=
        real[i] *
        w;


      norm[
        offset + i
      ] +=
        w *
        w;

    }


    // --------------------------------------------------------
    // Right
    // --------------------------------------------------------

    real.fill(0);
    imag.fill(0);


    for (
      let bin = 0;
      bin < DIM_F;
      bin++
    ) {

      const index =
        bin *
        DIM_T +
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


    reconstructConjugateSpectrum(
      real,
      imag
    );


    fft6144(
      real,
      imag,
      true
    );


    for (
      let i = 0;
      i < N_FFT;
      i++
    ) {

      right[
        offset + i
      ] +=
        real[i] *
        window[i];

    }

  }


  const outputLeft =
    new Float32Array(
      CHUNK_SIZE
    );

  const outputRight =
    new Float32Array(
      CHUNK_SIZE
    );


  for (
    let i = 0;
    i < CHUNK_SIZE;
    i++
  ) {

    const source =
      i +
      TRIM;


    const n =
      Math.max(
        norm[source],
        1e-7
      );


    outputLeft[i] =
      left[source] /
      n;


    outputRight[i] =
      right[source] /
      n;

  }


  return {

    left:
      outputLeft,

    right:
      outputRight

  };

}


// ============================================================
// Spectrum reconstruction
// ============================================================

function reconstructConjugateSpectrum(
  real,
  imag
) {

  /*
   * DIM_F = 3072
   *
   * n_fft/2 = 3072
   *
   * Nyquist bin 3072 is set to zero.
   *
   * The negative-frequency half is reconstructed.
   */

  real[DIM_F] = 0;
  imag[DIM_F] = 0;


  for (
    let bin = 1;
    bin < DIM_F;
    bin++
  ) {

    const mirror =
      N_FFT -
      bin;


    real[mirror] =
      real[bin];


    imag[mirror] =
      -imag[bin];

  }

}


// ============================================================
// 6144 FFT
// ============================================================

function fft6144(
  real,
  imag,
  inverse
) {

  /*
   * 6144 = 3 × 2048
   *
   * Split into three 2048-point FFTs.
   */

  const N =
    6144;

  const M =
    2048;


  const aReal =
    new Float32Array(M);

  const aImag =
    new Float32Array(M);

  const bReal =
    new Float32Array(M);

  const bImag =
    new Float32Array(M);

  const cReal =
    new Float32Array(M);

  const cImag =
    new Float32Array(M);


  for (
    let i = 0;
    i < M;
    i++
  ) {

    aReal[i] =
      real[
        i * 3
      ];

    aImag[i] =
      imag[
        i * 3
      ];


    bReal[i] =
      real[
        i * 3 + 1
      ];

    bImag[i] =
      imag[
        i * 3 + 1
      ];


    cReal[i] =
      real[
        i * 3 + 2
      ];

    cImag[i] =
      imag[
        i * 3 + 2
      ];

  }


  fft2048(
    aReal,
    aImag,
    inverse
  );


  fft2048(
    bReal,
    bImag,
    inverse
  );


  fft2048(
    cReal,
    cImag,
    inverse
  );


  const sign =
    inverse
      ? 1
      : -1;


  for (
    let k = 0;
    k < M;
    k++
  ) {

    const angle =
      sign *
      2 *
      Math.PI *
      k /
      N;


    const w1r =
      Math.cos(angle);

    const w1i =
      Math.sin(angle);


    const w2r =
      Math.cos(
        angle * 2
      );

    const w2i =
      Math.sin(
        angle * 2
      );


    const br =
      bReal[k] *
      w1r -
      bImag[k] *
      w1i;


    const bi =
      bReal[k] *
      w1i +
      bImag[k] *
      w1r;


    const cr =
      cReal[k] *
      w2r -
      cImag[k] *
      w2i;


    const ci =
      cReal[k] *
      w2i +
      cImag[k] *
      w2r;


    const x0r =
      aReal[k] +
      br +
      cr;


    const x0i =
      aImag[k] +
      bi +
      ci;


    const x1r =
      aReal[k] +
      br * COS120 -
      bi * SIN120 +
      cr * COS240 -
      ci * SIN240;


    const x1i =
      aImag[k] +
      br * SIN120 +
      bi * COS120 +
      cr * SIN240 +
      ci * COS240;


    const x2r =
      aReal[k] +
      br * COS240 -
      bi * SIN240 +
      cr * COS120 -
      ci * SIN120;


    const x2i =
      aImag[k] +
      br * SIN240 +
      bi * COS240 +
      cr * SIN120 +
      ci * COS120;


    real[k] =
      x0r;

    imag[k] =
      x0i;


    real[k + M] =
      x1r;

    imag[k + M] =
      x1i;


    real[k + M * 2] =
      x2r;

    imag[k + M * 2] =
      x2i;

  }


  if (inverse) {

    for (
      let i = 0;
      i < N;
      i++
    ) {

      real[i] /=
        N;

      imag[i] /=
        N;

    }

  }

}


const COS120 =
  -0.5;

const SIN120 =
  Math.sqrt(3) / 2;

const COS240 =
  -0.5;

const SIN240 =
  -Math.sqrt(3) / 2;


// ============================================================
// 2048 FFT
// ============================================================

function fft2048(
  real,
  imag,
  inverse
) {

  const N =
    2048;


  // Bit reversal
  let j = 0;


  for (
    let i = 1;
    i < N;
    i++
  ) {

    let bit =
      N >> 1;


    while (
      j & bit
    ) {

      j ^=
        bit;

      bit >>=
        1;

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
    len <= N;
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


    const wr0 =
      Math.cos(angle);

    const wi0 =
      Math.sin(angle);


    const half =
      len >> 1;


    for (
      let start = 0;
      start < N;
      start += len
    ) {

      let wr = 1;
      let wi = 0;


      for (
        let k = 0;
        k < half;
        k++
      ) {

        const i =
          start +
          k;

        const p =
          i +
          half;


        const vr =
          real[p] *
          wr -
          imag[p] *
          wi;


        const vi =
          real[p] *
          wi +
          imag[p] *
          wr;


        const ur =
          real[i];

        const ui =
          imag[i];


        real[i] =
          ur +
          vr;

        imag[i] =
          ui +
          vi;


        real[p] =
          ur -
          vr;

        imag[p] =
          ui -
          vi;


        const nextWr =
          wr *
          wr0 -
          wi *
          wi0;


        wi =
          wr *
          wi0 +
          wi *
          wr0;


        wr =
          nextWr;

      }

    }

  }

}


// ============================================================
// Hann
// ============================================================

function makeHannWindow(
  size
) {

  const window =
    new Float32Array(
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
// Blend window
// ============================================================

function createBlendWindow(
  length,
  overlap
) {

  const window =
    new Float32Array(
      length
    );


  window.fill(1);


  const fade =
    Math.min(
      overlap,
      Math.floor(
        length / 2
      )
    );


  for (
    let i = 0;
    i < fade;
    i++
  ) {

    const x =
      i /
      fade;


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
// Audio → MIDI
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
    const note of timedNotes
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
            note.amplitude ??
            0.8
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


  const rendered =
    await offline.startRendering();


  return rendered;

}


// ============================================================
// MIDI normalization
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
// Merge same pitch
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
    const note of sorted
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
// Remove short notes
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
// Fill small gaps
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