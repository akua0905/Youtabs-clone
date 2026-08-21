import * as tf from "@tensorflow/tfjs";
import {
  BasicPitch,
  outputToNotesPoly,
  addPitchBendsToNoteEvents,
  noteFramesToTime
} from "@spotify/basic-pitch";
import { Midi } from "@tonejs/midi";

const input = document.getElementById("audio");
const button = document.getElementById("convert");
const status = document.getElementById("status");
const download = document.getElementById("download");

const MODEL_URL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";

let selectedFile = null;

input.addEventListener("change", () => {
  selectedFile = input.files?.[0] ?? null;
  download.style.display = "none";

  if (!selectedFile) {
    status.textContent = "音源を選択してください。";
    return;
  }

  status.textContent =
    `選択済み\n${selectedFile.name}`;
});

button.addEventListener("click", async () => {
  try {
    if (!selectedFile) {
      status.textContent =
        "先に音源を選択してください。";
      return;
    }

    button.disabled = true;
    download.style.display = "none";

    status.textContent =
      "音源を読み込んでいます……";

    const arrayBuffer =
      await selectedFile.arrayBuffer();

    const audioContext =
      new AudioContext();

    const originalBuffer =
      await audioContext.decodeAudioData(arrayBuffer);

    status.textContent =
      `音源読み込み完了\n` +
      `元のサンプルレート: ${originalBuffer.sampleRate} Hz\n` +
      `長さ: ${originalBuffer.duration.toFixed(1)}秒\n\n` +
      `22,050 Hzへ変換しています……`;

    const audioBuffer =
      await resampleAudio(
        originalBuffer,
        22050
      );

    await audioContext.close();

    status.textContent =
      "AIモデルを読み込んでいます……";

    await tf.setBackend("webgl");
    await tf.ready();

    const basicPitch =
      new BasicPitch(MODEL_URL);

    const frames = [];
    const onsets = [];
    const contours = [];

    status.textContent =
      "AI解析中……\n0%";

    await basicPitch.evaluateModel(
      audioBuffer,

      (f, o, c) => {
        frames.push(...f);
        onsets.push(...o);
        contours.push(...c);
      },

      (progress) => {
        status.textContent =
          `AI解析中……\n${Math.round(progress * 100)}%`;
      }
    );

    status.textContent =
      "音符を抽出しています……";

    const notes =
      outputToNotesPoly(
        frames,
        onsets,
        0.25,
        0.25,
        5
      );

    const notesWithBends =
      addPitchBendsToNoteEvents(
        contours,
        notes
      );

    let timedNotes =
      noteFramesToTime(
        notesWithBends
      );

    if (!timedNotes.length) {
      throw new Error(
        "音符を検出できませんでした。"
      );
    }

    // =========================================
    // MIDI用ノート整理
    // =========================================

    const beforeCount =
      timedNotes.length;

    timedNotes =
      mergeSamePitchNotes(timedNotes);

    timedNotes =
      removeTinyNotes(timedNotes);

    timedNotes =
      fillTinyGaps(timedNotes);

    const afterCount =
      timedNotes.length;

    status.textContent =
      `音符を整理しています……\n` +
      `${beforeCount}音 → ${afterCount}音`;

    // =========================================
    // MIDI生成
    // =========================================

    const midi =
      new Midi();

    const track =
      midi.addTrack();

    track.name =
      "Basic Pitch";

    for (const note of timedNotes) {

      track.addNote({
        midi: note.pitchMidi,
        time: note.startTimeSeconds,
        duration: note.durationSeconds,
        velocity: Math.max(
          0.01,
          Math.min(
            1,
            note.amplitude ?? 0.8
          )
        )
      });

    }

    const midiData =
      midi.toArray();

    const blob =
      new Blob(
        [midiData],
        { type: "audio/midi" }
      );

    const url =
      URL.createObjectURL(blob);

    const filename =
      selectedFile.name
        .replace(/\.[^/.]+$/, "") +
      ".mid";

    download.href = url;
    download.download = filename;

    download.textContent =
      `MIDIを保存（${timedNotes.length}音）`;

    download.style.display =
      "block";

    status.textContent =
      `解析完了\n\n` +
      `解析前: ${beforeCount}音\n` +
      `整理後: ${afterCount}音\n` +
      `ファイル: ${filename}`;

  } catch (error) {

    console.error(error);

    status.textContent =
      `エラーが発生しました。\n\n` +
      `${error.name}: ${error.message}`;

  } finally {

    button.disabled = false;

  }
});


// ======================================================
// 同じ音高の連続ノートを結合
// ======================================================

function mergeSamePitchNotes(notes) {

  if (notes.length <= 1) {
    return notes;
  }

  const sorted =
    [...notes].sort(
      (a, b) =>
        a.startTimeSeconds -
        b.startTimeSeconds
    );

  const result = [];

  // これ以下の隙間なら「同じ音」とみなして結合
  const MERGE_GAP = 0.08;

  for (const note of sorted) {

    const previous =
      result[result.length - 1];

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
      previous.pitchMidi ===
      note.pitchMidi;

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

      // 音量は大きい方を採用
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


// ======================================================
// 極端に短いノートを削除
// ======================================================

function removeTinyNotes(notes) {

  // 80ms未満の音を削除
  const MIN_DURATION = 0.08;

  return notes.filter(
    note =>
      note.durationSeconds >=
      MIN_DURATION
  );
}


// ======================================================
// ごく短い隙間を埋める
// ======================================================

function fillTinyGaps(notes) {

  if (notes.length <= 1) {
    return notes;
  }

  const MAX_GAP = 0.05;

  const sorted =
    [...notes].sort(
      (a, b) =>
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
      current.pitchMidi !==
      next.pitchMidi
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
      gap <= MAX_GAP
    ) {

      current.durationSeconds +=
        gap;

    }
  }

  return sorted;
}


// ======================================================
// 48kHzなどの音源を22.05kHzへ変換
// ======================================================

async function resampleAudio(
  sourceBuffer,
  targetSampleRate
) {

  const numberOfChannels =
    sourceBuffer.numberOfChannels;

  const targetLength =
    Math.ceil(
      sourceBuffer.duration *
      targetSampleRate
    );

  const offlineContext =
    new OfflineAudioContext(
      numberOfChannels,
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

  return await offlineContext.startRendering();
}