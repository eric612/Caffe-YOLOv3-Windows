#convert_box_data.cpp -> io.cpp
##image
datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),buf.size()));
datum->set_encoded(true);

##label
datum->add_float_data(float(label));
datum->add_float_data(float(difficult));
datum->add_float_data(box[i]);

#box_data_layer.cpp
##Reshape()
int label_size = sides_[i] * sides_[i] * (30 * 5);

##load_batch()
nothing edit

##transform_label()
need to edit!

#base_data_layer.cpp
##LayerSetUp()
box_label_=false; //not edit.


