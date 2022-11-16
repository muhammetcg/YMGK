import 'package:flutter/material.dart';
//import 'package:mobys_deneme/main.dart';
// ignore: depend_on_referenced_packages
//import 'package:navigators/navigators.dart';
import 'package:mobys_deneme/Views/login.dart';

class NavBar extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          
          UserAccountsDrawerHeader(
            accountName: Text('MOBYS',
            style: TextStyle(
              fontSize: 25,
              color: Colors.white70,
            ),
            ),
            accountEmail: Text('example@gmail.com'),
            currentAccountPicture: CircleAvatar(
              child: ClipOval(
                child: Image.asset(
                  "images/Desd.gif",
                  width: 90,
                  height: 90,
                  fit: BoxFit.cover,
                ),

              ),
            ),

            decoration: BoxDecoration(
              //color: Colors.blue,
              image: DecorationImage(
                image: NetworkImage(
                  'https://granices.com/wp-content/uploads/2019/03/elaz%C4%B1%C4%9F.jpg',

                ),
                fit: BoxFit.cover,
                ),
            ),
         ),
          ListTile(
            leading: Icon(Icons.description),
            title: Text('Katalog'),
            //onTap: () => print('Fav'),
            onTap: () => null,
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.login),
            title: Text('Giriş Ekranı'),
            //onTap: () => print('Fav'),
            onTap: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => LoginView()));
            },
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.people),
            title: Text('Hakkımızda'),
            //onTap: () => print('Fav'),
            onTap: () => null,
          ),
          Divider(),
          ListTile(
            leading: Icon(Icons.contact_phone),
            title: Text('İletişim'),
            //onTap: () => print('Fav'),
            onTap: () => null,
          ),
        ],
      ),

    );
  }
}