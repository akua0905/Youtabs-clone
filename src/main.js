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

const DEMUCS_MODEL =
  "https://huggingface.co/StemSplitio/htdemucs-ft-vocals-onnx/resolve/main/htdemucs_ft_vocals_fp16weights.onnx";

const BASIC_PITCH_MODEL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";

const DEMUCS_SAMPLE_RATE = 44100;
const BASIC_PITCH_SAMPLE_RATE = 22050;

const DEMUCS_SAMPLES = 343980;
const DEMUCS_OVERLAP = Math.floor(DEMUCS_SAMPLES / 4);
const DEMUCS_STRIDE =
  DEMUCS_SAMPLES - DEMUCS_OVERLAP;

const VOCALS_STEM_ROW = 3;

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

  if (statusElement) {
    statusElement.textContent = message;
  }

}


function setProgress(value) {

  if (!progressBar) {
    return;
  }

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

    if (downloadElement) {
      downloadElement.style.display =
        "none";
    }


    let demucsSession = null;


    try {

      // ======================================================
      // 1. 音源読み込み
      // ======================================================

      setStatus(
        "音源を読み込んでいます……"
      );

      setProgress(0.01);


      const fileBuffer =
        await selectedFile.arrayBuffer();


      const decodedAudio =
        await decodeAudio(
          fileBuffer
        );


      if (
        !decodedAudio ||
        decodedAudio.length <= 0
      ) {

        throw new Error(
          "音源を正常に読み込めませんでした。"
        );

      }


      // ======================================================
      // 2. 44.1kHz Stereo
      // ======================================================

      setStatus(
        "音源を44.1kHzステレオに変換しています……"
      );

      setProgress(0.04);


      const demucsAudio =
        await convertToStereo44100(
          decodedAudio
        );


      const originalLeft =
        demucsAudio.left;

      const originalRight =
        demucsAudio.right;


      // ======================================================
      // 3. Demucs読み込み
      // ======================================================

      setStatus(
        "音源分離AIを読み込んでいます……\n" +
        "初回はモデルの読み込みに時間がかかります。"
      );

      setProgress(0.06);


      demucsSession =
        await createDemucsSession();


      // ======================================================
      // 4. Demucsモデル仕様確認
      // ======================================================

      validateDemucsModel(
        demucsSession
      );


      // ======================================================
      // 5. ボーカル抽出
      // ======================================================

      setStatus(
        "ボーカル／伴奏を分離しています……"
      );

      setProgress(0.08);


      const vocals =
        await extractVocals(
          demucsSession,
          originalLeft,
          originalRight,
          progress => {

            setProgress(
              0.08 +
              progress * 0.47
            );

            setStatus(
              "ボーカル／伴奏を分離しています……\n" +
              `${Math.round(progress * 100)}%`
            );

          }
        );


      // ======================================================
      // 6. 伴奏生成
      // ======================================================

      setStatus(
        "伴奏トラックを生成しています……"
      );

      setProgress(0.57);


      const instrumental =
        subtractVocals(
          originalLeft,
          originalRight,
          vocals.left,
          vocals.right
        );


      // ======================================================
      // 7. Demucs解放
      // ======================================================

      try {

        if (
          demucsSession &&
          typeof demucsSession.release ===
            "function"
        ) {

          await demucsSession.release();

        }

      } catch {}

      demucsSession = null;


      await waitForBrowser();


      // ======================================================
      // 8. Basic Pitch
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
      // 9. ボーカル → MIDI
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
              progress * 0.16
            );

          }
        );


      await waitForBrowser();


      // ======================================================
      // 10. 伴奏 → MIDI
      // ======================================================

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
              progress * 0.16
            );

          }
        );


      await waitForBrowser();


      // ======================================================
      // 11. ZIP
      // ======================================================

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
        "YouTabs Clone Error:",
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

      if (demucsSession) {

        try {

          if (
            typeof demucsSession.release ===
              "function"
          ) {

            await demucsSession.release();

          }

        } catch {}

      }

      convertButton.disabled =
        false;

    }

  }
);


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
    DEMUCS_SAMPLE_RATE;


  const targetLength =
    Math.max(
      1,
      Math.ceil(
        source.duration *
        targetRate
      )
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
// Demucs Session
// ============================================================

async function createDemucsSession() {

  ort.env.wasm.simd = true;

  ort.env.wasm.numThreads = 1;

  ort.env.wasm.proxy = false;


  const providers =
    ["wasm"];


  console.log(
    "ONNX Runtime:",
    ort.env
  );


  console.log(
    "Demucs model:",
    DEMUCS_MODEL
  );


  console.log(
    "Creating Demucs session..."
  );


  const session =
    await ort.InferenceSession.create(
      DEMUCS_MODEL,
      {
        executionProviders:
          providers,

        graphOptimizationLevel:
          "basic",

        enableCpuMemArena:
          false,

        enableMemPattern:
          false
      }
    );


  console.log(
    "Demucs input names:",
    session.inputNames
  );


  console.log(
    "Demucs input metadata:",
    session.inputMetadata
  );


  console.log(
    "Demucs output names:",
    session.outputNames
  );


  console.log(
    "Demucs output metadata:",
    session.outputMetadata
  );


  return session;

}


// ============================================================
// Demucsモデル仕様確認
// ============================================================

function validateDemucsModel(
  session
) {

  if (
    !session.inputNames ||
    session.inputNames.length === 0
  ) {

    throw new Error(
      "Demucsモデルに入力が存在しません。"
    );

  }


  const inputName =
    session.inputNames[0];


  const metadata =
    session.inputMetadata?.[inputName];


  console.log(
    "Actual Demucs input:",
    inputName,
    metadata
  );


  /*
    正しいモデル:

    name:
      mix

    dimensions:
      [1, 2, 343980]

    dtype:
      float32
  */


  if (
    inputName !== "mix"
  ) {

    throw new Error(
      "読み込んだDemucsモデルの入力名が想定と異なります。\n\n" +
      `実際: ${inputName}\n` +
      "期待: mix\n\n" +
      "モデルURLまたはモデルファイルが正しくありません。"
    );

  }


  const dimensions =
    metadata?.dimensions;


  if (
    !dimensions ||
    dimensions.length !== 3
  ) {

    throw new Error(
      "Demucsモデルの入力次元数が異なります。\n\n" +
      `実際: ${JSON.stringify(dimensions)}\n` +
      "期待: [1, 2, 343980]\n\n" +
      "別のONNXモデルを読み込んでいる可能性があります。"
    );

  }


  const expected =
    [
      1,
      2,
      DEMUCS_SAMPLES
    ];


  for (
    let i = 0;
    i < expected.length;
    i++
  ) {

    const actual =
      dimensions[i];


    /*
      ONNX Runtimeでは
      dynamic dimensionが
      null / undefined / -1
      になる場合があるため、
      その場合は固定値チェックを
      スキップする。
    */

    if (
      actual === null ||
      actual === undefined ||
      actual === -1 ||
      actual === "?"
    ) {

      continue;

    }


    if (
      Number(actual) !==
      expected[i]
    ) {

      throw new Error(
        "Demucsモデルの入力形状が想定と違います。\n\n" +
        `実際: ${JSON.stringify(dimensions)}\n` +
        `期待: ${JSON.stringify(expected)}\n\n` +
        "現在のモデルURLが正しいHT-Demucsモデルを指しているか確認してください。"
      );

    }

  }


  console.log(
    "Demucs model shape OK:",
    dimensions
  );

}


// ============================================================
// Vocal Extraction
// ============================================================

async function extractVocals(
  session,
  left,
  right,
  onProgress
) {

  const total =
    left.length;


  const vocalsLeft =
    new Float32Array(
      total
    );


  const vocalsRight =
    new Float32Array(
      total
    );


  const weight =
    new Float32Array(
      total
    );


  const window =
    makeTransitionWindow(
      DEMUCS_SAMPLES,
      DEMUCS_OVERLAP
    );


  const chunkBuffer =
    new Float32Array(
      2 *
      DEMUCS_SAMPLES
    );


  const chunkCount =
    Math.max(
      1,
      Math.ceil(
        total /
        DEMUCS_STRIDE
      )
    );


  const inputName =
    session.inputNames[0];


  const outputName =
    session.outputNames.includes(
      "stems"
    )
      ? "stems"
      : session.outputNames[0];


  for (
    let chunkIndex = 0;
    chunkIndex < chunkCount;
    chunkIndex++
  ) {

    const start =
      chunkIndex *
      DEMUCS_STRIDE;


    const end =
      Math.min(
        start +
        DEMUCS_SAMPLES,
        total
      );


    const chunkLength =
      end -
      start;


    chunkBuffer.fill(0);


    chunkBuffer
      .subarray(
        0,
        chunkLength
      )
      .set(
        left.subarray(
          start,
          end
        )
      );


    chunkBuffer
      .subarray(
        DEMUCS_SAMPLES,
        DEMUCS_SAMPLES +
        chunkLength
      )
      .set(
        right.subarray(
          start,
          end
        )
      );


    /*
      ここが重要。

      HT-Demucs FT vocals ONNX:

      [1, 2, 343980]

      44.1kHz stereo
    */

    const inputTensor =
      new ort.Tensor(
        "float32",
        chunkBuffer,
        [
          1,
          2,
          DEMUCS_SAMPLES
        ]
      );


    /*
      実行前にshapeを確認。
    */

    console.log(
      "Running Demucs:",
      {
        inputName,
        shape:
          inputTensor.dims,
        length:
          inputTensor.data.length
      }
    );


    let result;


    try {

      result =
        await session.run({
          [inputName]:
            inputTensor
        });

    } catch (error) {

      throw new Error(
        "Demucsの推論に失敗しました。\n\n" +
        (
          error instanceof Error
            ? error.message
            : String(error)
        ) +
        "\n\n" +
        "入力shape: [1, 2, 343980]"
      );

    }


    const output =
      result[outputName];


    if (!output) {

      throw new Error(
        "Demucsの出力が見つかりません。\n\n" +
        `出力名: ${session.outputNames.join(", ")}`
      );

    }


    console.log(
      "Demucs output:",
      output.dims
    );


    /*
      正しい出力:

      [1, 4, 2, 343980]
    */

    if (
      !output.dims ||
      output.dims.length !== 4
    ) {

      throw new Error(
        "Demucsの出力shapeが想定と違います。\n\n" +
        `実際: ${JSON.stringify(output.dims)}\n` +
        "期待: [1, 4, 2, 343980]"
      );

    }


    const data =
      output.data;


    const channels =
      2;


    const samples =
      DEMUCS_SAMPLES;


    const stemCount =
      4;


    const rowOffset =
      VOCALS_STEM_ROW *
      channels *
      samples;


    for (
      let s = 0;
      s < chunkLength;
      s++
    ) {

      const w =
        window[s];


      const vocalLeft =
        data[
          rowOffset +
          s
        ];


      const vocalRight =
        data[
          rowOffset +
          samples +
          s
        ];


      vocalsLeft[
        start + s
      ] +=
        vocalLeft *
        w;


      vocalsRight[
        start + s
      ] +=
        vocalRight *
        w;


      weight[
        start + s
      ] +=
        w;

    }


    onProgress(
      (chunkIndex + 1) /
      chunkCount
    );


    /*
      Safariに制御を返す。
    */

    await waitForBrowser();

  }


  // ==========================================================
  // Overlap-add正規化
  // ==========================================================

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


    vocalsLeft[i] /=
      w;


    vocalsRight[i] /=
      w;

  }


  return {
    left:
      vocalsLeft,

    right:
      vocalsRight
  };

}


// ============================================================
// Instrumental
// ============================================================

function subtractVocals(
  originalLeft,
  originalRight,
  vocalsLeft,
  vocalsRight
) {

  const length =
    originalLeft.length;


  const instrumentalLeft =
    new Float32Array(
      length
    );


  const instrumentalRight =
    new Float32Array(
      length
    );


  for (
    let i = 0;
    i < length;
    i++
  ) {

    instrumentalLeft[i] =
      originalLeft[i] -
      vocalsLeft[i];


    instrumentalRight[i] =
      originalRight[i] -
      vocalsRight[i];

  }


  return {
    left:
      instrumentalLeft,

    right:
      instrumentalRight
  };

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


  /*
    MIDI tempoは120 BPM固定。

    ノートのtime/durationは秒ベースで
    Basic Pitchの実時間を維持する。
  */

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
    DEMUCS_SAMPLE_RATE;


  const targetRate =
    BASIC_PITCH_SAMPLE_RATE;


  const length =
    stereo.left.length;


  const mono =
    new Float32Array(
      length
    );


  /*
    ステレオ → モノラル。

    ここでは左右を平均する。
  */

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
    Math.max(
      1,
      Math.ceil(
        sourceBuffer.duration *
        targetRate
      )
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
// 小さい隙間を埋める
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


// ============================================================
// Demucs Transition Window
// ============================================================

function makeTransitionWindow(
  segment,
  overlap
) {

  const window =
    new Float32Array(
      segment
    );


  window.fill(1);


  for (
    let i = 0;
    i < overlap;
    i++
  ) {

    const value =
      i /
      overlap;


    window[i] =
      value;


    window[
      segment -
      1 -
      i
    ] =
      value;

  }


  return window;

}