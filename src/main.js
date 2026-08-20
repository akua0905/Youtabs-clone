import * as tf from "@tensorflow/tfjs";
import {
  BasicPitch,
  addPitchBendsToNoteEvents,
  noteFramesToTime,
  outputToNotesPoly
} from "@spotify/basic-pitch";
import { Midi } from "@tonejs/midi";

const input = document.getElementById("audio");
const button = document.getElementById("convert");
const status = document.getElementById("status");
const download = document.getElementById("download");

const MODEL_URL =
  "https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/model/model.json";

let audioFile = null;

input.addEventListener("change", () => {
  audioFile = input.files[0];

  download.style.display = "none";

  if (!audioFile) {
    status.textContent = "音源を選択してください。";
    return;
  }

  const sizeMB = (audioFile.size / 1024 / 1024).toFixed(1);

  status.textContent =
    `選択済み:\n${audioFile.name}\n${sizeMB} MB`;
});

button.addEventListener("click", async () => {
  if (!audioFile) {
    status.textContent = "先に音源を選択してください。";
    return;
  }

  button.disabled = true;
  download.style.display = "none";

  try {
    status.textContent = "音源を読み込んでいます……";

    const arrayBuffer = await audioFile.arrayBuffer();

    const audioContext = new AudioContext();

    const decoded =
      await audioContext.decodeAudioData(arrayBuffer);

    status.textContent = "音声を解析用形式に変換しています……";

    const audioData = await resampleToMono22050(
      decoded
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
      "MIDI解析を開始します……\n0%";

    await basicPitch.evaluateModel(
      audioData,

      (f, o, c) => {
        frames.push(...f);
        onsets.push(...o);
        contours.push(...c);
      },

      (percent) => {
        const value =
          Math.round(percent * 100);

        status.textContent =
          `MIDI解析中……\n${value}%`;
      }
    );

    status.textContent =
      "検出した音符をMIDIに変換しています……";

    const noteEvents =
      noteFramesToTime(
        addPitchBendsToNoteEvents(
          contours,
          outputToNotesPoly(
            frames,
            onsets,
            0.25,
            0.25,
            5
          )
        )
      );

    if (noteEvents.length === 0) {
      throw new Error(
        "音符を検出できませんでした。"
      );
    }

    const midi = new Midi();

    const track = midi.addTrack();

    track.name = "Basic Pitch";

    for (const note of noteEvents) {
      track.addNote({
        midi: note.pitchMidi,
        time: note.startTimeSeconds,
        duration: note.durationSeconds,
        velocity: Math.max(
          0.01,
          Math.min(1, note.amplitude)
        )
      });

      if (note.pitchBends) {
        note.pitchBends.forEach(
          (bend, index) => {
            track.addPitchBend({
              time:
                note.startTimeSeconds +
                (
                  note.durationSeconds *
                  index
                ) /
                note.pitchBends.length,
              value: bend
            });
          }
        );
      }
    }

    const midiData = midi.toArray();

    const blob = new Blob(
      [midiData],
      { type: "audio/midi" }
    );

    const url =
      URL.createObjectURL(blob);

    const originalName =
      audioFile.name.replace(
        /\.[^/.]+$/,
        ""
      );

    download.href = url;

    download.download =
      `${originalName}.mid`;

    download.textContent =
      `MIDIを保存（${noteEvents.length}音）`;

    download.style.display = "block";

    status.textContent =
      `解析完了。\n` +
      `検出音符: ${noteEvents.length}\n\n` +
      `「MIDIを保存」を押してください。`;

  } catch (error) {

    console.error(error);

    status.textContent =
      "エラーが発生しました。\n\n" +
      error.message;

  } finally {
    button.disabled = false;
  }
});


async function resampleToMono22050(
  audioBuffer
) {

  const targetRate = 22050;

  const length =
    Math.ceil(
      audioBuffer.duration *
      targetRate
    );

  const offline =
    new OfflineAudioContext(
      1,
      length,
      targetRate
    );

  const source =
    offline.createBufferSource();

  source.buffer = audioBuffer;

  const gain =
    offline.createGain();

  const channelCount =
    audioBuffer.numberOfChannels;

  const merger =
    offline.createChannelMerger(
      channelCount
    );

  source.connect(merger);

  for (
    let channel = 0;
    channel < channelCount;
    channel++
  ) {
    merger.connect(
      gain,
      channel,
      0
    );
  }

  gain.gain.value =
    1 / channelCount;

  gain.connect(
    offline.destination
  );

  source.start(0);

  const rendered =
    await offline.startRendering();

  return rendered.getChannelData(0);
}
