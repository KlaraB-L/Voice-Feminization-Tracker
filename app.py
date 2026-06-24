import streamlit as st
import sqlite3
import json
import os
import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt

# --- KONFIGURATION ---
DB_FILE = "stimm_rpg.db"
UPLOAD_DIR = "."  # Der Punkt bedeutet: Nutze einfach den Hauptordner!

PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
MAX_FORMANT_HZ = 5500.0
F1_CLOSE_MAX = 450.0
F1_OPEN_MIN = 750.0
MIN_FRAMES_PER_BIN = 8

LIT_PITCH = 200.0       
LIT_F2 = 2100.0
LIT_STD = 25.0

# --- DATENBANK FUNKTIONEN ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS idole (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            file_path TEXT,
            pitch_median REAL,
            pitch_std REAL,
            f2_global REAL,
            f2_bins TEXT,
            jitter REAL,
            shimmer REAL,
            hnr REAL
        )
    ''')
    conn.commit()
    conn.close()

def speichere_idol_in_db(name, file_path, features):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO idole 
            (name, file_path, pitch_median, pitch_std, f2_global, f2_bins, jitter, shimmer, hnr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, file_path, 
            features["pitch_median"], features["pitch_std"], features["f2_global"],
            json.dumps(features["f2_bins"]), features["jitter"], features["shimmer"], features["hnr"]
        ))
        conn.commit()
    except Exception as e:
        st.error(f"Datenbankfehler: {e}")
    finally:
        conn.close()

def lade_alle_idole():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT name, file_path, pitch_median, pitch_std, f2_global, f2_bins, jitter, shimmer, hnr FROM idole')
    rows = cursor.fetchall()
    conn.close()
    
    idole = {}
    for r in rows:
        idole[r[0]] = {
            "file_path": r[1], "pitch_median": r[2], "pitch_std": r[3], "f2_global": r[4],
            "f2_bins": json.loads(r[5]), "jitter": r[6], "shimmer": r[7], "hnr": r[8]
        }
    return idole

# --- AKUSTISCHE ANALYSE (Aus deinem Skript) ---
def analysiere_stimme(sound):
    ergebnis = {"valid": False}
    try:
        pitch_obj = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        intensity_obj = sound.to_intensity()
        formant_obj = sound.to_formant_burg(time_step=None, max_number_of_formants=5, maximum_formant=MAX_FORMANT_HZ)
        
        zeiten = pitch_obj.xs()
        alle_intensitaeten = [intensity_obj.get_value(t) for t in zeiten if intensity_obj.get_value(t) is not None and not np.isnan(intensity_obj.get_value(t))]
        if not alle_intensitaeten: return ergebnis
        schwellenwert = max(alle_intensitaeten) - 25.0

        f0_w, f1_w, f2_w = [], [], []
        for t in zeiten:
            f0 = pitch_obj.get_value_at_time(t)
            if f0 is None or np.isnan(f0) or f0 <= 0: continue
            if intensity_obj.get_value(t) < schwellenwert: continue
            f1, f2 = formant_obj.get_value_at_time(1, t), formant_obj.get_value_at_time(2, t)
            if f1 is None or f2 is None or np.isnan(f1) or np.isnan(f2): continue
            f0_w.append(f0)
            f1_w.append(f1)
            f2_w.append(f2)

        if not f0_w: return ergebnis
        f0_arr, f1_arr, f2_arr = np.array(f0_w), np.array(f1_w), np.array(f2_w)
        
        bins = {"geschlossen": [], "mittel": [], "offen": []}
        for f1, f2 in zip(f1_arr, f2_arr):
            if f1 < F1_CLOSE_MAX: bins["geschlossen"].append(f2)
            elif f1 > F1_OPEN_MIN: bins["offen"].append(f2)
            else: bins["mittel"].append(f2)

        bin_stats = {}
        for k, v in bins.items():
            if len(v) >= MIN_FRAMES_PER_BIN: bin_stats[k] = {"f2_mean": float(np.mean(v)), "n": len(v)}

        point_process = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
        harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=PITCH_FLOOR)
        hnr_w = harmonicity.values.flatten()
        hnr = float(np.mean(hnr_w[hnr_w > -200])) if hnr_w.size > 0 else np.nan

        return {
            "valid": True, "pitch_median": float(np.median(f0_arr)), "pitch_std": float(np.std(f0_arr)),
            "f2_global": float(np.mean(f2_arr)), "f2_bins": bin_stats, "jitter": jitter, "shimmer": shimmer, "hnr": hnr
        }
    except Exception:
        return ergebnis

# --- VERGLEICHS LOGIK ---
def berechne_scores(live, idol):
    p_diff = abs(idol["pitch_median"] - live["pitch_median"])
    p_score = max(0, min(100, int(100 - (p_diff * 2.5))))
    
    # Resonanz
    g_bins = [b for b in idol["f2_bins"] if b in live["f2_bins"]]
    if g_bins:
        sc, gw = [], []
        for b in g_bins:
            if live["f2_bins"][b]["f2_mean"] >= idol["f2_bins"][b]["f2_mean"]: sc.append(100)
            else: sc.append(max(0, min(100, int(100 - (idol["f2_bins"][b]["f2_mean"] - live["f2_bins"][b]["f2_mean"]) * 0.6))))
            gw.append(min(idol["f2_bins"][b]["n"], live["f2_bins"][b]["n"]))
        r_score = int(np.average(sc, weights=gw))
    else:
        r_score = max(0, min(100, int(100 - abs(idol["f2_global"] - live["f2_global"]) * 0.6)))

    pros_score = 100 if live["pitch_std"] >= idol["pitch_std"] else max(0, min(100, int(100 - (idol["pitch_std"] - live["pitch_std"]) * 5.0)))
    
    # Quality
    sq_sc = []
    if not np.isnan(live["jitter"]): sq_sc.append(100 if live["jitter"] <= idol["jitter"] else max(0, int(100 - (live["jitter"] - idol["jitter"]) * 30)))
    if not np.isnan(live["shimmer"]): sq_sc.append(100 if live["shimmer"] <= idol["shimmer"] else max(0, int(100 - (live["shimmer"] - idol["shimmer"]) * 15)))
    if not np.isnan(live["hnr"]): sq_sc.append(100 if live["hnr"] >= idol["hnr"] else max(0, int(100 - (idol["hnr"] - live["hnr"]) * 5)))
    sq_score = int(np.mean(sq_sc)) if sq_sc else 50

    gesamt = int((p_score * 0.30) + (r_score * 0.35) + (pros_score * 0.15) + (sq_score * 0.20))
    return gesamt, p_score, r_score, pros_score, sq_score

# --- APP STRUKTUR ---
init_db()
st.set_page_config(page_title="Stimm-RPG v2 Web", layout="wide")
st.title("🎙️ Stimm-RPG v2: Logopädisches Web-Training")

# SIDEBAR: Datenbank & Verwaltung
st.sidebar.header("📁 Referenz-Datenbank (Idole)")
alle_idole = lade_alle_idole()

# 1. Neues Idol hochladen
with st.sidebar.expander("➕ Neues Idol hinzufügen"):
    neu_name = st.text_input("Name des Idols")
    neu_file = st.file_uploader("WAV-Datei hochladen", type=["wav"])
    if st.button("Idol analysieren & speichern") and neu_name and neu_file:
        p = os.path.join(UPLOAD_DIR, neu_file.name)
        with open(p, "wb") as f: f.write(neu_file.getbuffer())
        with st.spinner("Analysiere Vorbild..."):
            features = analysiere_stimme(parselmouth.Sound(p))
            if features["valid"]:
                speichere_idol_in_db(neu_name, p, features)
                st.success(f"'{neu_name}' erfolgreich gespeichert!")
                st.rerun()
            else:
                st.error("Audio ungültig oder keine Stimme erkannt.")

# 2. Vorbild aussuchen & Vorkost hören
if alle_idole:
    ausgewaehltes_idol_name = st.sidebar.selectbox("Wähle dein Trainings-Idol:", list(alle_idole.keys()))
    idol_daten = alle_idole[ausgewaehltes_idol_name]
    
    st.sidebar.write("🎧 **Hörprobe / Vorkost:**")
    if os.path.exists(idol_daten["file_path"]):
        st.sidebar.audio(idol_daten["file_path"], format="audio/wav")
    else:
        st.sidebar.warning("Audiodatei lokal nicht gefunden.")
else:
    st.sidebar.info("Bitte lade zuerst ein Idol hoch.")
    st.stop()

# HAUPTSEITE: Aufnahme & Training
st.subheader("📜 Phonetisch ausbalancierter Trainingssatz:")
st.info('“Während Yvonne früh am Ufer über glühende Kohlen schritt und dabei laut über ihre Lieblingsbücher, große Öko-Häuser und ungewöhnliche Pflanzenarten nachdachte...”')

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### 🎤 Deine Aufnahme")
    # Streamlits nativer Audio-Recorder (Nutzt das eingebaute iPhone/PC Mikrofon!)
    live_audio_file = st.audio_input("Klicke zum Aufnehmen")

if live_audio_file and idol_daten:
    with open("temp_live.wav", "wb") as f: f.write(live_audio_file.getbuffer())
    live_features = analysiere_stimme(parselmouth.Sound("temp_live.wav"))
    
    if live_features["valid"]:
        gesamt, p_sc, r_sc, pr_sc, q_sc = berechne_scores(live_features, idol_daten)
        
        with col2:
            st.write(f"### 🏆 Gesamt-Match: {gesamt}%")
            st.progress(gesamt / 100)
            
            # --- DIAGRAMME GENERIEREN ---
            fig, axs = plt.subplots(4, 1, figsize=(6, 8))
            plt.style.use('dark_background')
            
            # Pitch
            axs[0].barh(['Deine Tonhöhe'], [live_features["pitch_median"]], color='#3498db', height=0.4)
            axs[0].axvline(idol_daten["pitch_median"], color='white', linestyle='--', label=f'Idol ({idol_daten["pitch_median"]:.0f} Hz)')
            axs[0].set_xlim(0, 300)
            axs[0].legend(loc='upper right', fontsize=8)
            
            # Resonanz
            g_bins = [b for b in idol_daten["f2_bins"] if b in live_features["f2_bins"]]
            if g_bins:
                y_pos = np.arange(len(g_bins))
                axs[1].barh(y_pos - 0.2, [idol_daten["f2_bins"][b]["f2_mean"] for b in g_bins], height=0.35, color='white', alpha=0.4, label='Idol')
                axs[1].barh(y_pos + 0.2, [live_features["f2_bins"][b]["f2_mean"] for b in g_bins], height=0.35, color='#9b59b6', label='Du')
                axs[1].set_yticks(y_pos)
                axs[1].set_yticklabels([b.capitalize() for b in g_bins])
            axs[1].set_xlim(0, 2800)
            axs[1].legend(loc='upper right', fontsize=8)
            
            # Melodie
            axs[2].barh(['Stimm-Melodie'], [live_features["pitch_std"]], color='#e67e22', height=0.4)
            axs[2].axvline(idol_daten["pitch_std"], color='white', linestyle='--', label=f'Idol ({idol_daten["pitch_std"]:.1f} Hz)')
            axs[2].set_xlim(0, 80)
            axs[2].legend(loc='upper right', fontsize=8)
            
            # Qualität (Aufgeteilt in Jitter, Shimmer, HNR)
            sq_labels = ['Jitter (%)', 'Shimmer (%)', 'HNR (dB)']
            y_pos_sq = np.arange(len(sq_labels))
            axs[3].barh(y_pos_sq - 0.2, [idol_daten["jitter"], idol_daten["shimmer"], idol_daten["hnr"]], height=0.35, color='white', alpha=0.4, label='Idol')
            axs[3].barh(y_pos_sq + 0.2, [live_features["jitter"], live_features["shimmer"], live_features["hnr"]], height=0.35, color='#2ecc71', label='Du')
            axs[3].set_yticks(y_pos_sq)
            axs[3].set_yticklabels(sq_labels)
            axs[3].set_xlim(0, max(35, live_features["hnr"]+5, idol_daten["hnr"]+5))
            axs[3].legend(loc='upper right', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Es konnte keine valide Stimme in deiner Aufnahme analysiert werden. Sprich lauter oder klarer!")
