package MOBILE.AI.IT.phraserecognitionbasedtranslatorforkids;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;

import MOBILE.AI.IT.phraserecognitionbasedtranslatorforkids.classifier.Classifier;

public class MainActivity extends AppCompatActivity {

    private Bitmap bitmap;
    public Classifier classifier = new Classifier();
    ImageView imageView;
    Uri imageuri;
    Button buclassify;
    TextView classitext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView=(ImageView)findViewById(R.id.image); //imageView will show the chosen picture
        buclassify=(Button)findViewById(R.id.classify); //this is the classify button
        classitext=(TextView)findViewById(R.id.classifytext); //TextView will show the classification result

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent=new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Picture"),12);
            }
        });

        //Initializing classifier object that will produce predictions from pictures
        classifier.activity = this;
        classifier.initialize();

        buclassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //When clicked on Classify button:
                classifier.classify(bitmap); // Classifies the picture(bitmap) using classifier class
                showresult(); // Shows the results on TextView
            }
        });
    }

    private void showresult(){
        // Change the text of TextView to result of classifier prediction.
        classitext.setText(classifier.predict());
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==12 && resultCode==RESULT_OK && data!=null) {
            imageuri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageuri); //Reading the image from Media
                imageView.setImageBitmap(bitmap); // Insert image "bitmap" to imageView
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}