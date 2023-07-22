(ns heartbeats.core
  (:require [clojure.java.io :as io]
            [libpython-clj2.python :as py]
            [libpython-clj2.require :refer [require-python]]))

(require-python '[builtins :as pyb])
(require-python '[torch :as t])
(require-python '[torchaudio :as ta])
(require-python '[numpy :as np])
(require-python '[matplotlib.pyplot :as plt])
(require-python '[sounddevice :as sd])

(defn shape [array]
  (pyb/tuple (py/py.- array shape)))

(defn nullify-keyword [x]
  (if (keyword? x) nil x))

(defn index-spec->pyslice [idx]
  (cond
    (number? idx) idx
    (keyword? idx) (pyb/slice nil)
    (vector? idx) (apply pyb/slice (map nullify-keyword idx))))

(defn slice [array indices]
  (py/get-item array (mapv index-spec->pyslice indices)))

(defn plot
  ([xs show?]
   (if (sequential? xs)
     (do
       (doseq [x xs]
         (plot x false)))
     (let [s (shape wave)]
       (condp = (count s)
         1 (plt/plot xs)
         2 (doseq [x xs]
             (plt/plot x))
         :else (throw "Bad wave shape"))))
   (when show?
     (plt/show)))
  ([xs]
   (plot xs true)))

(defn normalize [wave]
  (let [ma (np/amax wave)
        mi (np/amin wave)]
    (-> (np/subtract wave mi)
        (np/divide (- ma mi))
        (np/multiply 2)
        (np/subtract 1))))

(defn threshold
  ([wave lower upper]
   (np/add (np/where (np/less wave lower)    -1 0)
           (np/where (np/greater wave upper) +1 0)))
  ([wave limit]
   (assert (> limit 0))
   (threshold wave (- limit) limit)))

(def audio-file-name "emily_hope_heartbeat_lizs_heart_after_light_exercise_003.mp3")
(def audio-file (.getFile (io/resource audio-file-name)))
(def wave-and-rate (ta/load audio-file))
(def wave (first wave-and-rate))
(def rate (second wave-and-rate))

;; Convert 2 channel audio to 1 channel using averaging
(def wave (np/mean (py/py. wave numpy) 0))

;; Listen to the sound
(sd/play wave :samplerate rate)

;; Detect and plot heartbeats on the first second of the audio clip and save to disk
(def first-second (slice (normalize wave) [[0 rate]]))
(plot [(threshold first-second 0.2) first-second] false)
(plt/savefig "heartbeats.pdf")
