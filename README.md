# machineL
Σημείωση
Όλα τα scripts κατεβάζουν τα CSVs από το github.Για να το αλλάξετε πρέπει να κάνετε uncomment την μεταβλητή σε 
κάθε αρχείο και τότε θα το ανοίγει απο τον ίδιο φάκελο που βρίσκεται το script.

SCRIPT1
Ενώνει τα 2 CSVs
Κάνει annotate τα highlevel classes με βάση τα sub-classes
Κάνει sample με βάση τα minority low level classes.Εδώ το έκανα με το χέρι βλέποντας τα counts και υπολόγισα
πόσα χρειάζονται απο κάθε κλάσση.
Σώζει το CSV

SCRIPT2
Ανοίγει το CSV από το προηγούμενο και κάνει τα ακόλουθα
Normalize όλα τα columns που δεν είναι στο range[0,1] επιλέχθηκαν με το χέρι
Hash trick στις τιμές του column service και drop 
One hot encoding στο protocol type και drop
Μετατροπή του flag σε binary και drop
Σώζει το csv

SCRIPT3
Ελένχει 3 σενάρια και δίνει τα αποτελέσματα
Σενάριο πρώτο είναι fit SVC/DT στο dataset από το προηγούμενο
Σενάριο δεύτερο fit SVC/DT στο dataset από το προηγούμενο με class weights
Σενάριο τρίτο fit SVC/DT στο dataset από το προηγούμενο με smote
Σχετικά με την σειρά των κλάσεων στα αποτελέσματα στο confusion matrix είναι πάντα με την ίδια σειρά

SCRIPT4
Ανοίγει το CSV κάνει smote για όλες τις κλάσεις και σώζει το CSV

SCRIPT5
Ανοίγει το csv απο το προηγούμενο
και κάνει fit στον DT
εξάγει τα σημαντικότερα features
και δημιουργεί ένα plot με το score του κάθε ενός
Σώζει το csv

SCRIPT6
Ανοίγει το csv απο το προηγούμενο
και δημιουργεί το correlation matrix
υπολογίζει το άθροισμα των απόλυτων τιμών της συσχέτισης του κάθε feature με όλα τα άλλα
Κρατάει τα 20 πιο ασυσχέτιστα με βάση το παραπάνω score
Σώζει το csv

SCRIPT7
Ανοίγει ένα απο τα 2 προηγούμενα CSVs και τρέχει τον SVC/DT 
Δηλώνει τα αποτελέσματα

SCRIPT8
Ανοίγει το CSV από το script6
κάνει kFold cross validation για τον DT
GridSearchCV για τις 4 πιο σημαντικές hyperparams
Αφού εξάγει το καλύτερο μοντέλο υπολογίζει το average recall
Αν είναι καλύτερο απο το προηγούμενο το σώζει
Σώζει το καλύτερο μοντέλο

SCRIPT9
Ανοίγει ένα CSV με λίγα samples που είχα αφήσει εκτός 
φορτώνει τον classifier απο το προηγούμενο βήμα
και κάνει το prediction
Δηλώνει τα αποτελέσματα

SCRIPT10
Φορτώνει τον classifier
Ανοίγει το csv με τα 5000 samples
κάνει predict
ενημερώνει για τον χρόνο που πήρε

