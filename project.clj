(defproject heartbeats "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [clj-python/libpython-clj "2.024"]]
  :repl-options {:init-ns heartbeats.core}
  :jvm-opts ["--add-modules" "jdk.incubator.foreign,jdk.incubator.vector"
             "--enable-native-access=ALL-UNNAMED"
             "-Djdk.attach.allowAttachSelf"])
