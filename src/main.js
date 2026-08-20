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
      status.textContent = "先に音源を選択してください。";
      return;
    }

    button.disabled = true;
    download.style.display = "none";

    status.textContent = "音源を読み込んでいます……";

    const arrayBuffer =
      await selectedFile.arrayBuffer();

    const audioContext =
      new AudioContext();

    const audioBuffer =
      await audioContext.decodeAudioData(arrayBuffer);

    status.textContent =
      `音源読み込み完了\n` +
      `長さ: ${audioBuffer.duration.toFixed(1)}秒\n\n` +
      `AIモデルを読み込んでいます……`;

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

    const timedNotes =
      noteFramesToTime(
        notesWithBends
      );

    if (!timedNotes.length) {
      throw new Error(
        "音符を検出できませんでした。"
      );
    }

    status.textContent =
      `MIDIを生成しています……\n` +
      `${timedNotes.length}個の音符を検出`;

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
        velocity: note.amplitude
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
        .replace(/\.[^/.]+$/, "")
        + ".mid";

    download.href = url;
    download.download = filename;
    download.textContent =
      `MIDIを保存（${timedNotes.length}音）`;

    download.style.display =
      "block";

    status.textContent =
      `解析完了\n\n` +
      `検出音符: ${timedNotes.length}\n` +
      `ファイル: ${filename}`;

    await audioContext.close();

  } catch (error) {

    console.error(error);

    status.textContent =
      "エラーが発生しました。\n\n" +
      `${error.name}: ${error.message}`;

  } finally {
    button.disabled = false;
  }
});