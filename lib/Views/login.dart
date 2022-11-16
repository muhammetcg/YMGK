
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:mobys_deneme/Views/sign-in.dart';
import 'package:mobys_deneme/common/theme_helper.dart';
import 'package:mobys_deneme/widgets/header_widget.dart';

class LoginView extends StatefulWidget {
  const LoginView({Key? key}) : super(key: key);

  @override
  _LoginViewState createState() => _LoginViewState();
  
}

class _LoginViewState extends State<LoginView>{
  double _headerHeight = 250;
  Key _formKey = GlobalKey<FormState>();
  @override
  Widget build(BuildContext context) {
   return Scaffold(
    backgroundColor: Colors.white70,
    body: SingleChildScrollView(
      child: Column(
        children: [
          Container(
            height: _headerHeight,
            child: HeaderWidget(_headerHeight, true, Icons.login_rounded),
          ),
          SafeArea(
            child: Container(
              padding: EdgeInsets.fromLTRB(20, 10, 20, 10),
              margin: EdgeInsets.fromLTRB(20, 10, 20, 10),
              child: Column(
                children: [
                  Text(
                    'MOBYS',
                    style: TextStyle(fontSize: 60, fontWeight: FontWeight.bold)
                  ),
                  Text(
                    'Signin into your account',
                    style: TextStyle(color: Colors.grey)
                  ),
                  SizedBox(height: 30.0),
                  Form(
                    key: _formKey,
                    child: Column(
                      children: [
                        TextField(
                          decoration: TehemeHelper().textInputDecoration('User Name',  'Enter your user name',),
                        ),
                        SizedBox(height: 30.0,),
                        TextField(
                          obscureText: true ,
                          decoration: TehemeHelper().textInputDecoration( 'Password', 'Enter your password') ,
                        ),
                        SizedBox(height: 15.0,),
                        Container(
                          margin: EdgeInsets.fromLTRB(10, 0, 10, 20 ),
                          alignment: Alignment.topRight,
                          child: Text('Forgot Your Password?'),
                        ),
                        Container(
                          decoration:TehemeHelper().buttonBoxDecoration(context),
                          child: ElevatedButton(
                            style: TehemeHelper().buttonStyle(),
                            
                            child: Padding(
                              padding: EdgeInsets.fromLTRB(40,10, 40, 10),
                              child: Text('Sign In'.toUpperCase(), style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white ),),
                            ),
                            onPressed: () {
                              
                              },
                          ),

                        ),
                        Container(
                           margin: EdgeInsets.fromLTRB(10, 20, 10, 20 ),
                          //child: Text('Dont have an account? Create '),
                          child: Text.rich(
                            TextSpan(
                              children: [
                                TextSpan(text: "Dont have an account?"),
                                TextSpan(
                                  text: 'Create',
                                  recognizer: TapGestureRecognizer()
                                  ..onTap = (){
                                    Navigator.push(context, MaterialPageRoute(builder: (context) => SigninPage()));
                                  },
                                  // ignore: deprecated_member_use
                                  style: TextStyle(fontWeight: FontWeight.bold, color: Theme.of(context).accentColor ),
                                ),
                              ]
                            )
                          ),
                        ),
                      ],

                    ),

                  ),
                ],
              ),
              


          ),
          ),
        ],

      ),

    ),
   );
  }

}