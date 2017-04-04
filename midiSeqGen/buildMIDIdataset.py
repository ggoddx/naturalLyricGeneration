import midi, os


def main():
    rootdir = './130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Metal_Rock_rock.freemidis.net_MIDIRip/midi/i/iron_maiden/'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            pattern = midi.read_midifile(rootdir + file)

            bass = midi.Pattern()

            drum = midi.Pattern()

            for track in pattern:
                for event in track:
                    if type(event) == midi.events.TrackNameEvent:
                        trackName = event.text.lower()

                        if trackName.find('bass') != -1:
                            bass.append(track)

                        if trackName.find('drum') != -1:
                            drum.append(track)

            midi.write_midifile('./bassMIDI/bass_' + file, bass)
            midi.write_midifile('./drumMIDI/drum_' + file, drum)

    return

if __name__ == '__main__':
    main()
