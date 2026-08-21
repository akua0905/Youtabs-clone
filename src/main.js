import * as tf from "@tensorflow/tfjs";

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

const MODEL_URL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";

// MIDIテンポ
// 後で自動BPM検出に変更可能
const MIDI_BPM = 120;

// 同じ音高を結合する最大の隙間
const MERGE_GAP = 0.08;

// 極端に短い音を削除
const MIN_DURATION = 0.08;

// 同じ音高の小さな隙間を埋める
const MAX_FILL_GAP = 0.05;


// ============================================================
// HTML要素
// ============================================================

const input =
  document.getElementById("audio");

const button =
  document.getElementById("convert");

const status =
  document.getElementById("status");

const download =
  document.getElementById("download");


// ============================================================
// 状態
// ============================================================

let selectedFile = null;


// ============================================================
// ファイル選択
// ============================================================

input.addEventListener("change", () => {

  selectedFile =
    input.files?.[0] ?? null;

  download.style.display =
    "none";

  if (!selectedFile) {

    status.textContent =
      "音源を選択してください。";

    return;
  }

  const sizeMB =
    (
      selectedFile.size /
      1024 /
      1024
    ).toFixed(1);

  status.textContent =
    `選択済み\n` +
    `${selectedFile.name}\n` +
    `${sizeMB} MB`;

});


// ============================================================
// MIDI変換
// ============================================================

button.addEventListener(
  "click",
  async () => {

    let audioContext = null;

    try {

      // --------------------------------------------------------
      // ファイル確認
      // --------------------------------------------------------

      if (!selectedFile) {

        status.textContent =
          "先に音源を選択してください。";

        return;
      }


      // --------------------------------------------------------
      // UI
      // --------------------------------------------------------

      button.disabled = true;

      download.style.display =
        "none";

      status.textContent =
        "音源を読み込んでいます……";


      // --------------------------------------------------------
      // ファイル読み込み
      // --------------------------------------------------------

      const arrayBuffer =
        await selectedFile.arrayBuffer();


      // --------------------------------------------------------
      // AudioContext
      // --------------------------------------------------------

      audioContext =
        new AudioContext();


      const originalBuffer =
        await audioContext.decodeAudioData(
          arrayBuffer
        );


      status.textContent =
        `音源読み込み完了\n\n` +
        `サンプルレート: ` +
        `${originalBuffer.sampleRate} Hz\n` +
        `チャンネル: ` +
        `${originalBuffer.numberOfChannels}ch\n` +
        `長さ: ` +
        `${originalBuffer.duration.toFixed(1)}秒\n\n` +
        `22,050 Hzへ変換しています……`;


      // --------------------------------------------------------
      // 22,050Hzへ変換
      // --------------------------------------------------------

      const audioBuffer =
        await resampleAudio(
          originalBuffer,
          22050
        );


      // 元AudioContextを閉じる
      await audioContext.close();

      audioContext = null;


      // --------------------------------------------------------
      // TensorFlow.js
      // --------------------------------------------------------

      status.textContent =
        "AIモデルを準備しています……";


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


      // --------------------------------------------------------
      // Basic Pitch
      // --------------------------------------------------------

      status.textContent =
        "AIモデルを読み込んでいます……";


      const basicPitch =
        new BasicPitch(
          MODEL_URL
        );


      const frames = [];
      const onsets = [];
      const contours = [];


      status.textContent =
        "AI解析中……\n0%";


      // --------------------------------------------------------
      // Basic Pitch解析
      // --------------------------------------------------------

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

          const percent =
            Math.round(
              progress * 100
            );

          status.textContent =
            `AI解析中……\n${percent}%`;

        }

      );


      // --------------------------------------------------------
      // ノート抽出
      // --------------------------------------------------------

      status.textContent =
        "音符を抽出しています……";


      const notes =
        outputToNotesPoly(

          frames,
          onsets,

          // onset threshold
          0.25,

          // frame threshold
          0.25,

          // minimum note length
          5

        );


      // --------------------------------------------------------
      // Pitch Bend
      // --------------------------------------------------------

      const notesWithBends =
        addPitchBendsToNoteEvents(

          contours,
          notes

        );


      // --------------------------------------------------------
      // フレーム → 秒
      // --------------------------------------------------------

      let timedNotes =
        noteFramesToTime(
          notesWithBends
        );


      if (
        !timedNotes ||
        timedNotes.length === 0
      ) {

        throw new Error(
          "音符を検出できませんでした。"
        );

      }


      // --------------------------------------------------------
      // 採譜結果を整理
      // --------------------------------------------------------

      const beforeCount =
        timedNotes.length;


      status.textContent =
        `音符を整理しています……\n` +
        `${beforeCount}音`;


      // 同じ音高の連続ノートを結合
      timedNotes =
        mergeSamePitchNotes(
          timedNotes
        );


      // 極端に短いノートを削除
      timedNotes =
        removeTinyNotes(
          timedNotes
        );


      // 小さな隙間を補正
      timedNotes =
        fillTinyGaps(
          timedNotes
        );


      // 時間順に並べ直す
      timedNotes.sort(
        (
          a,
          b
        ) =>
          a.startTimeSeconds -
          b.startTimeSeconds
      );


      const afterCount =
        timedNotes.length;


      // --------------------------------------------------------
      // MIDI生成
      // --------------------------------------------------------

      status.textContent =
        `MIDIを生成しています……\n` +
        `${beforeCount}音 → ${afterCount}音`;


      const midi =
        new Midi();


      // --------------------------------------------------------
      // MIDIテンポ
      // --------------------------------------------------------

      midi.header.setTempo(
        MIDI_BPM
      );


      // --------------------------------------------------------
      // MIDIトラック
      // --------------------------------------------------------

      const track =
        midi.addTrack();


      track.name =
        "Basic Pitch";


      // --------------------------------------------------------
      // ノートを書き込み
      // --------------------------------------------------------

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
              note.amplitude ?? 0.8
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


      // --------------------------------------------------------
      // MIDI → Blob
      // --------------------------------------------------------

      const midiData =
        midi.toArray();


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


      // --------------------------------------------------------
      // ファイル名
      // --------------------------------------------------------

      const filename =
        selectedFile.name
          .replace(
            /\.[^/.]+$/,
            ""
          ) +
        ".mid";


      // --------------------------------------------------------
      // ダウンロードリンク
      // --------------------------------------------------------

      download.href =
        url;

      download.download =
        filename;

      download.textContent =
        `MIDIを保存（${afterCount}音）`;

      download.style.display =
        "block";


      // --------------------------------------------------------
      // 完了
      // --------------------------------------------------------

      status.textContent =
        `解析完了\n\n` +

        `元の音符数: ` +
        `${beforeCount}\n` +

        `整理後: ` +
        `${afterCount}\n` +

        `テンポ: ` +
        `${MIDI_BPM} BPM\n\n` +

        `ファイル: ` +
        `${filename}`;


    } catch (error) {

      // --------------------------------------------------------
      // エラー
      // --------------------------------------------------------

      console.error(
        "YouTabs Clone Error:",
        error
      );


      status.textContent =
        `エラーが発生しました。\n\n` +

        `${error.name || "Error"}: ` +

        `${error.message || error}`;


    } finally {

      // --------------------------------------------------------
      // 後処理
      // --------------------------------------------------------

      if (audioContext) {

        try {

          await audioContext.close();

        } catch {}

      }

      button.disabled =
        false;

    }

  }
);


// ============================================================
// 音声リサンプリング
// ============================================================

async function resampleAudio(

  sourceBuffer,

  targetSampleRate

) {

  const channels =
    sourceBuffer.numberOfChannels;


  const targetLength =
    Math.ceil(
      sourceBuffer.duration *
      targetSampleRate
    );


  const offlineContext =
    new OfflineAudioContext(

      channels,

      targetLength,

      targetSampleRate

    );


  const source =
    offlineContext.createBufferSource();


  source.buffer =
    sourceBuffer;


  source.connect(
    offlineContext.destination
  );


  source.start(0);


  const renderedBuffer =
    await offlineContext.startRendering();


  return renderedBuffer;

}


// ============================================================
// 同じ音高の連続ノートを結合
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
    const note of sorted
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


    const currentStart =
      note.startTimeSeconds;


    const gap =
      currentStart -
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

      const currentEnd =
        note.startTimeSeconds +
        note.durationSeconds;


      const newEnd =
        Math.max(
          previousEnd,
          currentEnd
        );


      previous.durationSeconds =
        newEnd -
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
// 短すぎるノートを削除
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


  for (
    let i = 0;
    i < sorted.length - 1;
    i++
  ) {

    const current =
      sorted[i];

    const next =
      sorted[i + 1];


    const samePitch =
      Math.round(
        current.pitchMidi
      ) ===
      Math.round(
        next.pitchMidi
      );


    if (!samePitch) {

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