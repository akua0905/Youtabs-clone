import {
  BasicPitch,
  outputToNotesPoly,
  addPitchBendsToNoteEvents,
  noteFramesToTime
} from "@spotify/basic-pitch";

import { Midi } from "@tonejs/midi";


// ============================================================
// 設定
// ============================================================

const BASIC_PITCH_MODEL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";

const BASIC_PITCH_SAMPLE_RATE = 22050;


// ------------------------------------------------------------
// MIDI後処理設定
// ------------------------------------------------------------

// これ未満のノートは誤検出として削除
const MIN_NOTE_DURATION = 0.07;

// 同じ音程でこの時間以内なら結合
const MERGE_GAP = 0.12;

// ほぼ重なっている同音ノートを統合
const OVERLAP_TOLERANCE = 0.05;

// 同一音程のノート間にある小さな隙間を埋める
const FILL_GAP = 0.08;

// 音量が極端に小さいノートを除去
const MIN_AMPLITUDE = 0.08;

// MIDIテンポ
const MIDI_BPM = 120;


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
    statusElement.textContent =
      message;
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


function yieldBrowser() {

  return new Promise(
    resolve =>
      setTimeout(
        resolve,
        0
      )
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

      downloadElement.removeAttribute(
        "href"
      );

    }


    setProgress(0);


    if (!selectedFile) {

      setStatus(
        "音源を選択してください。"
      );

      return;

    }


    const sizeMB =
      selectedFile.size /
      1024 /
      1024;


    setStatus(
      `選択済み\n${selectedFile.name}\n` +
      `${sizeMB.toFixed(1)} MB`
    );

  }
);


// ============================================================
// メイン
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


    convertButton.disabled =
      true;


    if (downloadElement) {

      downloadElement.style.display =
        "none";

    }


    try {

      // ------------------------------------------------------
      // 1. 音源読み込み
      // ------------------------------------------------------

      setStatus(
        "音源を読み込んでいます……"
      );

      setProgress(0.03);


      const arrayBuffer =
        await selectedFile.arrayBuffer();


      const decoded =
        await decodeAudio(
          arrayBuffer
        );


      if (
        !decoded ||
        decoded.length <= 0
      ) {

        throw new Error(
          "音源を読み込めませんでした。"
        );

      }


      // ------------------------------------------------------
      // 2. ステレオ → モノラル + 22050Hz
      // ------------------------------------------------------

      setStatus(
        "採譜用音声を生成しています……\n" +
        `${decoded.sampleRate} Hz → 22050 Hz`
      );

      setProgress(0.10);


      const audioBuffer =
        await createBasicPitchAudioBuffer(
          decoded
        );


      // ------------------------------------------------------
      // 3. Basic Pitch
      // ------------------------------------------------------

      setStatus(
        "Basic Pitchで採譜しています……"
      );

      setProgress(0.15);


      const basicPitch =
        new BasicPitch(
          BASIC_PITCH_MODEL
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

          setProgress(
            0.15 +
            progress * 0.55
          );


          setStatus(
            "Basic Pitchで採譜しています……\n" +
            `${Math.round(progress * 100)}%`
          );

        }

      );


      await yieldBrowser();


      // ------------------------------------------------------
      // 4. ノート生成
      // ------------------------------------------------------

      setStatus(
        "MIDIノートを整理しています……"
      );

      setProgress(0.72);


      let notes =
        outputToNotesPoly(
          frames,
          onsets,

          // frame threshold
          0.20,

          // onset threshold
          0.20,

          // minimum note length
          4
        );


      // ------------------------------------------------------
      // 5. Pitch Bend
      // ------------------------------------------------------

      notes =
        addPitchBendsToNoteEvents(
          contours,
          notes
        );


      // ------------------------------------------------------
      // 6. フレーム → 秒
      // ------------------------------------------------------

      let timedNotes =
        noteFramesToTime(
          notes
        );


      // ------------------------------------------------------
      // 7. 正規化
      // ------------------------------------------------------

      timedNotes =
        normalizeNotes(
          timedNotes
        );


      // ------------------------------------------------------
      // 8. 低振幅ノイズ削除
      // ------------------------------------------------------

      timedNotes =
        removeLowAmplitudeNotes(
          timedNotes
        );


      // ------------------------------------------------------
      // 9. 短すぎるノート削除
      // ------------------------------------------------------

      timedNotes =
        removeShortNotes(
          timedNotes
        );


      // ------------------------------------------------------
      // 10. 同音程ノート統合
      // ------------------------------------------------------

      timedNotes =
        mergeSamePitchNotes(
          timedNotes
        );


      // ------------------------------------------------------
      // 11. 小さな隙間を補正
      // ------------------------------------------------------

      timedNotes =
        fillSmallGaps(
          timedNotes
        );


      // ------------------------------------------------------
      // 12. 重複ノート整理
      // ------------------------------------------------------

      timedNotes =
        removeOverlappingSamePitchNotes(
          timedNotes
        );


      // ------------------------------------------------------
      // 13. 最終整理
      // ------------------------------------------------------

      timedNotes =
        finalCleanUp(
          timedNotes
        );


      // ------------------------------------------------------
      // 14. MIDI生成
      // ------------------------------------------------------

      setStatus(
        "MIDIファイルを生成しています……"
      );

      setProgress(0.88);


      const midi =
        createMidi(
          timedNotes
        );


      const midiData =
        midi.toArray();


      // ------------------------------------------------------
      // 15. Blob
      // ------------------------------------------------------

      const blob =
        new Blob(
          [midiData],
          {
            type:
              "audio/midi"
          }
        );


      const url =
        URL.createObjectURL(
          blob
        );


      const baseName =
        selectedFile.name
          .replace(
            /\.[^/.]+$/,
            ""
          );


      if (downloadElement) {

        downloadElement.href =
          url;

        downloadElement.download =
          `${baseName}_採譜.mid`;

        downloadElement.textContent =
          "MIDIを保存";

        downloadElement.style.display =
          "block";

      }


      setProgress(1);


      setStatus(
        "解析完了\n\n" +
        `検出ノート数：${timedNotes.length}\n\n` +
        "「MIDIを保存」から保存できます。"
      );


    } catch (error) {

      console.error(
        "Basic Pitch Error:",
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
// Basic Pitch用音声生成
// ============================================================

async function createBasicPitchAudioBuffer(
  source
) {

  const sourceRate =
    source.sampleRate;

  const targetRate =
    BASIC_PITCH_SAMPLE_RATE;


  const sourceLength =
    source.length;


  // ----------------------------------------------------------
  // ステレオ → モノラル
  // ----------------------------------------------------------

  const mono =
    new Float32Array(
      sourceLength
    );


  const channelCount =
    source.numberOfChannels;


  if (
    channelCount === 1
  ) {

    mono.set(
      source.getChannelData(0)
    );

  } else {

    const left =
      source.getChannelData(0);

    const right =
      source.getChannelData(
        1
      );


    for (
      let i = 0;
      i < sourceLength;
      i++
    ) {

      mono[i] =
        (
          left[i] +
          right[i]
        ) *
        0.5;

    }

  }


  // ----------------------------------------------------------
  // 元音声Buffer
  // ----------------------------------------------------------

  const sourceBuffer =
    new AudioBuffer({

      length:
        sourceLength,

      numberOfChannels:
        1,

      sampleRate:
        sourceRate

    });


  sourceBuffer
    .getChannelData(0)
    .set(mono);


  // ----------------------------------------------------------
  // 22050Hzへリサンプリング
  // ----------------------------------------------------------

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


  const node =
    offline.createBufferSource();


  node.buffer =
    sourceBuffer;


  node.connect(
    offline.destination
  );


  node.start(0);


  const rendered =
    await offline.startRendering();


  return rendered;

}


// ============================================================
// ノート正規化
// ============================================================

function normalizeNotes(
  notes
) {

  return notes

    .map(
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
    )

    .filter(
      note =>
        Number.isFinite(
          note.startTimeSeconds
        ) &&
        Number.isFinite(
          note.durationSeconds
        ) &&
        Number.isFinite(
          note.pitchMidi
        )
    );

}


// ============================================================
// 低振幅ノート削除
// ============================================================

function removeLowAmplitudeNotes(
  notes
) {

  return notes.filter(
    note =>
      note.amplitude >=
      MIN_AMPLITUDE
  );

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
// 同一音程のノート統合
// ============================================================

function mergeSamePitchNotes(
  notes
) {

  const sorted =
    [...notes].sort(
      (
        a,
        b
      ) => {

        if (
          a.pitchMidi !==
          b.pitchMidi
        ) {

          return (
            a.pitchMidi -
            b.pitchMidi
          );

        }

        return (
          a.startTimeSeconds -
          b.startTimeSeconds
        );

      }
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


    const samePitch =
      Math.round(
        previous.pitchMidi
      ) ===
      Math.round(
        note.pitchMidi
      );


    const previousEnd =
      previous.startTimeSeconds +
      previous.durationSeconds;


    const gap =
      note.startTimeSeconds -
      previousEnd;


    if (
      samePitch &&
      gap <= MERGE_GAP &&
      gap >= -OVERLAP_TOLERANCE
    ) {

      const noteEnd =
        note.startTimeSeconds +
        note.durationSeconds;


      previous.durationSeconds =
        Math.max(
          previousEnd,
          noteEnd
        ) -
        previous.startTimeSeconds;


      previous.amplitude =
        Math.max(
          previous.amplitude,
          note.amplitude
        );

    } else {

      result.push({
        ...note
      });

    }

  }


  return result.sort(
    (
      a,
      b
    ) =>
      a.startTimeSeconds -
      b.startTimeSeconds
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
      gap <= FILL_GAP
    ) {

      current.durationSeconds +=
        gap;

    }

  }


  return sorted;

}


// ============================================================
// 重複する同音ノートを整理
// ============================================================

function removeOverlappingSamePitchNotes(
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


  const result = [];


  for (
    const note
    of sorted
  ) {

    let absorbed =
      false;


    for (
      const previous
      of result
    ) {

      if (
        Math.round(
          previous.pitchMidi
        ) !==
        Math.round(
          note.pitchMidi
        )
      ) {

        continue;

      }


      const previousEnd =
        previous.startTimeSeconds +
        previous.durationSeconds;


      const noteEnd =
        note.startTimeSeconds +
        note.durationSeconds;


      const overlap =
        previousEnd -
        note.startTimeSeconds;


      if (
        overlap >= 0 &&
        overlap <=
          OVERLAP_TOLERANCE
      ) {

        previous.durationSeconds =
          Math.max(
            previousEnd,
            noteEnd
          ) -
          previous.startTimeSeconds;


        previous.amplitude =
          Math.max(
            previous.amplitude,
            note.amplitude
          );


        absorbed = true;

        break;

      }

    }


    if (!absorbed) {

      result.push({
        ...note
      });

    }

  }


  return result;

}


// ============================================================
// 最終クリーンアップ
// ============================================================

function finalCleanUp(
  notes
) {

  return notes

    .filter(
      note =>
        note.durationSeconds >=
        MIN_NOTE_DURATION
    )

    .map(
      note => ({

        ...note,

        pitchMidi:
          Math.max(
            0,
            Math.min(
              127,
              Math.round(
                note.pitchMidi
              )
            )
          ),

        startTimeSeconds:
          Math.max(
            0,
            note.startTimeSeconds
          ),

        durationSeconds:
          Math.max(
            0.01,
            note.durationSeconds
          ),

        amplitude:
          Math.max(
            0.01,
            Math.min(
              1,
              note.amplitude
            )
          )

      })
    )

    .sort(
      (
        a,
        b
      ) =>
        a.startTimeSeconds -
        b.startTimeSeconds
    );

}


// ============================================================
// MIDI生成
// ============================================================

function createMidi(
  notes
) {

  const midi =
    new Midi();


  /*
    120 BPMを基準にする。

    note.time / durationは
    秒としてTone.js MIDIへ渡す。
  */

  midi.header.setTempo(
    MIDI_BPM
  );


  const track =
    midi.addTrack();


  for (
    const note
    of notes
  ) {

    track.addNote({

      midi:
        note.pitchMidi,

      time:
        note.startTimeSeconds,

      duration:
        note.durationSeconds,

      velocity:
        note.amplitude

    });

  }


  return midi;

}