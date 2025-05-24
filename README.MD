Progetto testato utilizzando su python 3.12

Se il task del problema è trovare frasi corte in inglese e sbaglia tanto su questo potremmo pensare di aggiungere dati con frasi corte

Evitiamo di fare lo stemming

TODO
------------------------------------------------------------------
Dataset sbilanciato -> Testare sia con che senza undersampling

Aggiungere citazioni 

Mostrare Perchè logistic regression e NB
Capire multinomial NB


Creare docker

Usare widget python per mostrare le confusion matrix sul notebook

Check pipefile e generare requirements
Aggiungere -e . ai requirements
Creare readme 

Commentare notebook

Migliorare i commenti nei file di init del progetto

Installare pacchetto con pip install -e .


Considerare che la previsione funziona anche con parole non presenti nel vocabolario, ma potrebbe dare problemi di prestazioni
Il problema è che se il testo non ha nessuna parola presente nel dizionario, Verrà predetto Inglese di default

Check se pipeline tox funzionano da cold start

tox -e lint esegue ruff sia sul codice .py che nei notebook .ipynb
tox -e type-check esegue mypy sul codice e sui notebook
tox -e test esegue pytest