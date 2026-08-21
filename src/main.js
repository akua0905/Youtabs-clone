import * as ort from "onnxruntime-web";

import * as tf from "@tensorflow/tfjs";

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

const BASIC_PITCH_MODEL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";


const DEMUCS_MODEL =
  "https://huggingface.co/StemSplit/htdemucs/resolve/main/htdemucs_ft_vocals_fp16weights.onnx";


const DEMUCS_SAMPLE_RATE =
  44100;


const DEMUCS_SAMPLES =
  343980;


const DEMUCS_OVERLAP =
  Math.floor(DEMUCS_SAMPLES / 4);


const DEMUCS_STRIDE =
  DEMUCS_SAMPLES - DEMUCS_OVERLAP;


// MIDIテンポ
const MIDI_BPM = 120;


// ノート整理
const MERGE_GAP = 0.08;

const MIN_DURATION = 0.08;

const MAX_FILL_GAP = 0.05;


// ============================================================
// DOM
// ============================================================

const input =
  document.getElementById("audio");

const button =
  document.getElementById("convert");

const status =
  document.getElementById("status");

const download =
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

function setStatus(text) {

  status.textContent =
    text;

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


// ============================================================
// ファイル選択
// ============================================================

input.addEventListener(
  "change",
  () => {

    selectedFile =
      input.files?.[0] ?? null;

    download.style.display =
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

button.addEventListener(
  "click",
  async () => {

    if (!selectedFile) {

      setStatus(
        "先に音源を選択してください。"
      );

      return;
    }


    button.disabled =
      true;

    download.style.display =
      "none";


    try {

      // ------------------------------------------------------
      // 音源読み込み
      // ------------------------------------------------------

      setStatus(
        "音源を読み込んでいます……"
      );

      setProgress(0.02);


      const arrayBuffer =
        await selectedFile.arrayBuffer();


      const audioContext =
        new AudioContext();


      const decoded =
        await audioContext.decodeAudioData(
          arrayBuffer
        );


      await audioContext.close();


      // ------------------------------------------------------
      // 44.1kHz ステレオ化
      // ------------------------------------------------------

      setStatus(
        "音源を44.1kHzステレオに変換しています……"
      );

      setProgress(0.05);


      const stereo =
        await resampleStereo(
          decoded,
          DEMUCS_SAMPLE_RATE
        );


      // ------------------------------------------------------
      // Demucs
      // ------------------------------------------------------

      setStatus(
        "音源分離AIを準備しています……\n" +
        "初回は約166MBのモデルを読み込みます。"
      );

      setProgress(0.08);


      const session =
        await createDemucsSession();


      // ------------------------------------------------------
      // WebGPU / WASM
      // ------------------------------------------------------

      setStatus(
        "ボーカルと伴奏を分離しています……"
      );


      const separated =
        await separateDemucs(
          session,
          stereo.left,
          stereo.right,
          progress => {

            setProgress(
              0.08 +
              progress * 0.52
            );

            setStatus(
              `ボーカルと伴奏を分離しています……\n` +
              `${Math.round(progress * 100)}%`
            );

          }
        );


      // ------------------------------------------------------
      // Demucsを解放
      // ------------------------------------------------------

      try {

        await session.release();

      } catch {}


      // ------------------------------------------------------
      // Instrumental生成
      // ------------------------------------------------------

      setStatus(
        "伴奏トラックを生成しています……"
      );

      setProgress(0.62);


      const instrumental =
        createInstrumental(
          separated
        );


      // ------------------------------------------------------
      // Basic Pitch
      // ------------------------------------------------------

      setStatus(
        "Basic Pitchを準備しています……"
      );

      setProgress(0.64);


      try {

        await tf.setBackend(
          "webgl"
        );

      } catch {

        await tf.setBackend(
          "cpu"
        );

      }

      await tf.ready();


      const basicPitch =
        new BasicPitch(
          BASIC_PITCH_MODEL
        );


      // ------------------------------------------------------
      // ボーカル採譜
      // ------------------------------------------------------

      setStatus(
        "ボーカルをMIDIに変換しています……"
      );

      setProgress(0.66);


      const vocalsMidi =
        await analyzeToMidi(
          basicPitch,
          separated.vocals,
          progress => {

            setProgress(
              0.66 +
              progress * 0.16
            );

          }
        );


      // ------------------------------------------------------
      // 伴奏採譜
      // ------------------------------------------------------

      setStatus(
        "伴奏をMIDIに変換しています……"
      );

      setProgress(0.83);


      const instrumentalMidi =
        await analyzeToMidi(
          basicPitch,
          instrumental,
          progress => {

            setProgress(
              0.83 +
              progress * 0.12
            );

          }
        );


      // ------------------------------------------------------
      // ZIP
      // ------------------------------------------------------

      setStatus(
        "ZIPファイルを生成しています……"
      );

      setProgress(0.96);


      const zip =
        new JSZip();


      const baseName =
        selectedFile.name
          .replace(
            /\.[^/.]+$/,
            ""
          );


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


      download.href =
        url;


      download.download =
        `${baseName}_採譜.zip`;


      download.textContent =
        "ZIPを保存";


      download.style.display =
        "block";


      setProgress(1);


      setStatus(
        `解析完了\n\n` +
        `vocals.mid\n` +
        `instrumental.mid\n\n` +
        `ZIPにまとめました。`
      );


    } catch (error) {

      console.error(
        error
      );


      setStatus(
        `エラーが発生しました。\n\n` +
        `${error.name || "Error"}: ` +
        `${error.message || error}`
      );


      setProgress(0);

    } finally {

      button.disabled =
        false;

    }

  }
);


// ============================================================
// Demucsセッション
// ============================================================

async function createDemucsSession() {

  ort.env.wasm.numThreads =
    1;


  ort.env.wasm.simd =
    true;


  let executionProviders =
    ["wasm"];


  // WebGPUチェック
  if (
    "gpu" in navigator
  ) {

    try {

      const adapter =
        await navigator.gpu.requestAdapter();


      if (adapter) {

        executionProviders =
          ["webgpu", "wasm"];

      }

    } catch {

      console.log(
        "WebGPU unavailable."
      );

    }

  }


  console.log(
    "Execution providers:",
    executionProviders
  );


  const session =
    await ort.InferenceSession.create(
      DEMUCS_MODEL,
      {
        executionProviders,

        graphOptimizationLevel:
          "basic",

        enableCpuMemArena:
          false,

        enableMemPattern:
          false

      }
    );


  return session;

}


// ============================================================
// Demucs分離
// ============================================================

async function separateDemucs(

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
    Math.ceil(
      total /
      DEMUCS_STRIDE
    );


  for (
    let chunk = 0;
    chunk < chunkCount;
    chunk++
  ) {

    const start =
      chunk *
      DEMUCS_STRIDE;


    const end =
      Math.min(
        start +
        DEMUCS_SAMPLES,
        total
      );


    const length =
      end -
      start;


    chunkBuffer.fill(0);


    chunkBuffer
      .subarray(
        0,
        length
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
        length
      )
      .set(
        right.subarray(
          start,
          end
        )
      );


    const tensor =
      new ort.Tensor(
        "float32",
        chunkBuffer,
        [
          1,
          2,
          DEMUCS_SAMPLES
        ]
      );


    const result =
      await session.run({
        mix: tensor
      });


    const output =
      result.stems;


    if (!output) {

      throw new Error(
        "Demucsの出力テンソル 'stems' が見つかりません。"
      );

    }


    const data =
      output.data;


    /*
      Demucs 4-stem:

      0 = drums
      1 = bass
      2 = other
      3 = vocals

      shape:
      [1, 4, 2, samples]
    */


    const channels =
      2;


    const samples =
      DEMUCS_SAMPLES;


    const vocalRow =
      3;


    const instrumentalRows = [
      0,
      1,
      2
    ];


    for (
      let s = 0;
      s < length;
      s++
    ) {

      const w =
        window[s];


      const indexL =
        (
          vocalRow *
          channels *
          samples
        ) +
        s;


      const indexR =
        (
          vocalRow *
          channels *
          samples
        ) +
        (
          samples +
          s
        );


      vocalsLeft[
        start + s
      ] +=
        data[indexL] *
        w;


      vocalsRight[
        start + s
      ] +=
        data[indexR] *
        w;


      let instL = 0;
      let instR = 0;


      for (
        const row
        of instrumentalRows
      ) {

        const offset =
          row *
          channels *
          samples;


        instL +=
          data[
            offset + s
          ];


        instR +=
          data[
            offset +
            samples +
            s
          ];

      }


      instrumentalLeft[
        start + s
      ] +=
        instL *
        w;


      instrumentalRight[
        start + s
      ] +=
        instR *
        w;


      weight[
        start + s
      ] +=
        w;

    }


    onProgress(
      (chunk + 1) /
      chunkCount
    );


    // iPhoneのメモリ負荷を下げる
    await yieldToBrowser();

  }


  // overlap-add 正規化
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


    instrumentalLeft[i] /=
      w;


    instrumentalRight[i] /=
      w;

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
// ステレオ → 44.1kHz
// ============================================================

async function resampleStereo(

  sourceBuffer,

  targetRate

) {

  const targetLength =
    Math.ceil(
      sourceBuffer.duration *
      targetRate
    );


  const offline =
    new OfflineAudioContext(
      2,
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


  return {

    left:
      rendered
        .getChannelData(0)
        .slice(),

    right:
      (
        rendered.numberOfChannels > 1
          ? rendered.getChannelData(1)
          : rendered.getChannelData(0)
      ).slice()

  };

}


// ============================================================
// Instrumental生成
// ============================================================

function createInstrumental(
  separated
) {

  return {

    left:
      separated.instrumental.left,

    right:
      separated.instrumental.right

  };

}


// ============================================================
// Basic Pitch → MIDI
// ============================================================

async function analyzeToMidi(

  basicPitch,

  stereo,

  onProgress

) {

  // Basic Pitch用に22,050Hzへ
const audioBuffer =
  await createMonoAudioBuffer(
    stereo
  );


  const frames = [];
  const onsets = [];
  const contours = [];


  await basicPitch.evaluateModel(

    audioBuffer,

    (
      f,
      o,
      c
    ) => {

      frames.push(...f);
      onsets.push(...o);
      contours.push(...c);

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
    mergeSamePitchNotes(
      timedNotes
    );


  timedNotes =
    removeTinyNotes(
      timedNotes
    );


  timedNotes =
    fillTinyGaps(
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

    track.addNote({

      midi:
        Math.round(
          note.pitchMidi
        ),

      time:
        Math.max(
          0,
          note.startTimeSeconds
        ),

      duration:
        Math.max(
          0.01,
          note.durationSeconds
        ),

      velocity:
        Math.max(
          0.01,
          Math.min(
            1,
            note.amplitude ?? 0.8
          )
        )

    });

  }


  return midi.toArray();

}


// ============================================================
// Stereo → Mono 22,050Hz
// ============================================================

async function createMonoAudioBuffer(
  stereo
) {

  const length =
    stereo.left.length;


  const sampleRate =
    DEMUCS_SAMPLE_RATE;


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


  const targetRate =
    22050;


  const targetLength =
    Math.ceil(
      length *
      targetRate /
      sampleRate
    );


  const offline =
    new OfflineAudioContext(
      1,
      targetLength,
      targetRate
    );


  const buffer =
    offline.createBuffer(
      1,
      length,
      sampleRate
    );


  buffer
    .getChannelData(0)
    .set(mono);


  const source =
    offline.createBufferSource();


  source.buffer =
    buffer;


  source.connect(
    offline.destination
  );


  source.start(0);


  // OfflineAudioContextを同期的には返せないので、
  // 下のPromise版を使用する
  return resampleMonoBuffer(
    buffer,
    targetRate
  );

}


// ============================================================
// Mono resample
// ============================================================

async function resampleMonoBuffer(
  sourceBuffer,
  targetRate
) {

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
// 同音高ノート結合
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


  const result = [];


  for (
    const note
    of sorted
  ) {

    const previous =
      result[
        result.length - 1
      ];


    if (!previous) {

      result.push({
        ...note
      });

      continue;

    }


    const previousEnd =
      previous.startTimeSeconds +
      previous.durationSeconds;


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

      const end =
        Math.max(
          previousEnd,

          note.startTimeSeconds +
          note.durationSeconds
        );


      previous.durationSeconds =
        end -
        previous.startTimeSeconds;


      previous.amplitude =
        Math.max(
          previous.amplitude ?? 0,
          note.amplitude ?? 0
        );

    } else {

      result.push({
        ...note
      });

    }

  }


  return result;

}


// ============================================================
// 短すぎる音を削除
// ============================================================

function removeTinyNotes(
  notes
) {

  return notes.filter(
    note =>
      note.durationSeconds >=
      MIN_DURATION
  );

}


// ============================================================
// 小さな隙間を補正
// ============================================================

function fillTinyGaps(
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
// Demucs overlap window
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


// ============================================================
// iPhoneでUIを固めない
// ============================================================

function yieldToBrowser() {

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        0
      )
  );

}